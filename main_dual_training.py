import os
import sys
import uuid
from math import ceil
from typing import Union, List, Dict, Any
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from tqdm import trange
from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet, CifarNet
from src.advanced_optimizers import (
    Shampoo, Muon, Adam,
    MuonConfig, ShampooConfig, AdamConfig, OptimizerConfig,
    parse_optimizer_config, create_optimizer
)
from src.utils import convert_dataclasses
from src.dual_training_loggers import (
    create_default_loggers, LossAndAccuracyLogger,
    print_columns, print_training_details, compute_singular_values,
    DEFAULT_LOGGING_COLUMNS
)
from src.data import CifarLoader, CIFAR_MEAN, CIFAR_STD
from src.eval import evaluate
from src.param_groups import get_param_groups
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = True



ValidArch = Union[CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet, CifarNet]


def main(
    arch: ValidArch,
    optimizer_configs_str: List[str] = None,          
    optimizer_configs: List[OptimizerConfig] = None,  
    data_path: str = "cifar10",       
    batch_size: int = 2048,   
    lr_bias: float = 0.053,            
    lr_head: float = 0.67,          
    weight_decay: float = 1e-4,     
    weight_decay_misc: float = 1e-4,     
    batch_sweep_count: int = 20,
    use_augmentation: bool = True,    
    label_smoothing: float = 0.2,     
    device: str = "cuda",             
    seed: int = 0,                    
    save_results: bool = True,        
    svd_freq: int = 20,               
    total_train_steps: int = 400,     
):

    if optimizer_configs_str is not None:
        try:
            optimizer_configs = [parse_optimizer_config(s) for s in optimizer_configs_str]
        except Exception as e:
            print(f"Error parsing optimizer configs: {e}")
            print("\nExpected format: 'OptimizerType:param1=value1,param2=value2'")
            print("Examples:")
            print("  Muon:lr=0.0005,momentum=0.9")
            print("  Shampoo:lr=0.0005,momentum=0.9,order_multiplier=2")
            raise

    elif optimizer_configs is None:
        optimizer_configs = [
            ShampooConfig(lr=0.0005, momentum=0.9, order_multiplier=1),
            ShampooConfig(lr=0.0005, momentum=0.9, order_multiplier=2),
            MuonConfig(lr=0.0005, momentum=0.9),
        ]
    
    # Automatically determine parameter grouping strategy based on architecture
    arch_name = arch.__class__.__name__.lower()
    if arch_name == 'cifarnet':
        param_group_strategy = 'cifarnet'
    else:
        param_group_strategy = 'vit'  # Default for all other architectures
    
    print("=" * 80)
    print(f"Multi-Model Training: {len(optimizer_configs)} Optimizers")
    print("=" * 80)
    
    with open(sys.argv[0]) as f:
        code = f.read()
    config = convert_dataclasses({k: v for k, v in locals().items() 
                                   if k not in ['f', 'code']})
    config["cmd"] = " ".join(sys.argv)


    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    

    print("\n[1/5] Loading Data...")
    aug = dict(flip=True, translate=2) if use_augmentation else {}
    train_loader = CifarLoader(data_path, train=True, batch_size=batch_size, aug=aug)
    test_loader = CifarLoader(data_path, train=False, batch_size=2000)
    total_train_steps = ceil(batch_sweep_count * len(train_loader))
    total_epochs = ceil(total_train_steps / len(train_loader))
    print(f"  - Training samples: {len(train_loader.images)}")
    print(f"  - Test samples: {len(test_loader.images)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total steps: {total_train_steps}")
    print(f"  - Total epochs: {total_epochs}")
    # weight_decay_misc = weight_decay_misc * batch_size
    # weight_decay = weight_decay * batch_size
    
    print("\n[2/5] Creating Models...")
    models = {}
    base_model = arch.create(input_shape=(3, 32, 32), output_dim=10).to(device)
    
    # Initialize whitening layer for CifarNet
    if param_group_strategy == 'cifarnet' and hasattr(base_model, 'init_whiten'):
        print("  - Initializing whitening layer for CifarNet...")
        train_images = train_loader.normalize(train_loader.images[:5000])
        base_model.init_whiten(train_images)
    
    base_state_dict = base_model.state_dict()
    
    for i, opt_config in enumerate(optimizer_configs):
        model_name = f"{opt_config}"
        model = arch.create(input_shape=(3, 32, 32), output_dim=10).to(device)
        model.load_state_dict(base_state_dict)  
        models[model_name] = model
        print(f"  - Model {i+1}: {model_name}")
    
    print(f"  - Architecture: {arch.__class__.__name__}")
    print(f"  - Total parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"  - Number of models: {len(models)}")
    

    print("\n[3/5] Setting up Optimizers...")
    print(f"  - Using parameter grouping strategy: '{param_group_strategy}' (auto-detected from {arch.__class__.__name__})")
    
    # Calculate whitening bias training steps for CifarNet (3 epochs)
    whiten_bias_train_steps = ceil(3 * len(train_loader)) if param_group_strategy == 'cifarnet' else 0
    
    optimizers_dict = {}  
    filter_param_names_dict = {}  
    
    for model_name, (opt_config, model) in zip(models.keys(), zip(optimizer_configs, models.values())):

        # Use the parameter grouping strategy
        filter_params, head_params, bias_params, filter_names = get_param_groups(model, param_group_strategy)
        opts = create_optimizer(opt_config, filter_params, head_params, bias_params,
                               weight_decay, weight_decay_misc, lr_head, lr_bias)
        

        for opt in opts:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        optimizers_dict[model_name] = opts
        

        # Store filter param names for SVD tracking (must match filter_params)
        filter_param_names = filter_names
        filter_param_names_dict[model_name] = filter_param_names
        
        print(f"  - {model_name}: {len(filter_params)} filter, {len(head_params)} head, {len(bias_params)} bias params")
    

    model_logs = {name: [] for name in models.keys()}  
    singular_values_logs = {name: [] for name in models.keys()}  
    logger_data = {name: [] for name in models.keys()}  
    

    print("\n[4/5] Setting up Loggers...")
    logger_suite = create_default_loggers(
        label_smoothing=label_smoothing,
        track_singular_values=True  
    )
    process_loggers = logger_suite['process']
    group_loggers = logger_suite['group']
    print(f"  - Process loggers: {len(process_loggers)}")
    print(f"  - Group loggers: {len(group_loggers)}")
    

    print("\n[5/5] Training...")
    print_columns(DEFAULT_LOGGING_COLUMNS, is_head=True)
    
    step = 0

    
    for epoch in range(total_epochs):

        for model in models.values():
            model.train()
        

        epoch_metrics = {name: {'loss': 0.0, 'correct': 0, 'samples': 0} 
                        for name in models.keys()}
        

        for inputs, labels in train_loader:
            for model_name, model in models.items():

                # For CifarNet, control whitening bias gradient flow
                if param_group_strategy == 'cifarnet':
                    # Clone inputs to avoid in-place modifications from whitening layer
                    outputs = model(inputs.clone(), whiten_bias_grad=(step < whiten_bias_train_steps))
                else:
                    outputs = model(inputs)
                    
                loss = F.cross_entropy(outputs, labels, 
                                     label_smoothing=label_smoothing, reduction='sum')
                loss.backward()
                

                # Update learning rates
                opts = optimizers_dict[model_name]
                if param_group_strategy == 'cifarnet':
                    # For CifarNet: first optimizer is Adam with 3 param groups
                    # param_groups[0] = whitening bias (3 epoch schedule)
                    # param_groups[1] = other biases (standard schedule)
                    # param_groups[2] = head params (standard schedule)
                    adam_opt = opts[0]
                    adam_opt.param_groups[0]["lr"] = adam_opt.param_groups[0]["initial_lr"] * (1 - step / whiten_bias_train_steps) if step < whiten_bias_train_steps else 0.0
                    for group in adam_opt.param_groups[1:]:
                        group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                    
                    # Remaining optimizers (Shampoo/Muon for filters) use standard schedule
                    for opt in opts[1:]:
                        for group in opt.param_groups:
                            group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                else:
                    # Standard LR schedule for all optimizers
                    for opt in opts:
                        for group in opt.param_groups:
                            group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                

                for opt in optimizers_dict[model_name]:
                    opt.step()
                

                model.zero_grad(set_to_none=True)
                

                with torch.no_grad():
                    epoch_metrics[model_name]['loss'] += loss.item()
                    epoch_metrics[model_name]['correct'] += (outputs.argmax(1) == labels).float().sum().item()
                    epoch_metrics[model_name]['samples'] += len(inputs)
            

            if step % svd_freq == 0:
                print(f"\n  [Step {step}] Computing singular values...")
                for model_name, model in models.items():
                    sv = compute_singular_values(model, filter_param_names_dict[model_name])
                    singular_values_logs[model_name].append((step, sv))
                print(f"  [Step {step}] Recorded SVD for {len(models)} models")
            
            step += 1
            if step >= total_train_steps:
                break
        

        group_log_data = {}
        for logger in group_loggers:
            group_log_data.update(logger.log(models))
        

        for i, (model_name, model) in enumerate(models.items()):

            metrics = epoch_metrics[model_name]
            train_loss = metrics['loss'] / metrics['samples']
            train_acc = metrics['correct'] / metrics['samples']
            

            test_loss, test_acc = evaluate(model, test_loader)
            

            current_lr = optimizers_dict[model_name][0].param_groups[0]['lr'] if optimizers_dict[model_name] else 0.0
            

            process_log_data = {}
            for logger in process_loggers:
                log_output = logger.log(
                    model=model,
                    optimizer=optimizers_dict[model_name][0] if optimizers_dict[model_name] else None,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    epoch=epoch,
                    step=step,
                    total_steps=total_train_steps,
                )
                process_log_data.update(log_output)
            

            log_dict = {
                'epoch': epoch,
                'opt': model_name[:15],  
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
            }
            is_last = (i == len(models) - 1) and (epoch == total_epochs - 1)
            print_training_details(log_dict, is_final_entry=is_last)
            

            model_logs[model_name].append({
                'epoch': epoch,
                'step': step,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
            })
            

            detailed_log = {
                'epoch': epoch,
                'step': step,
                **process_log_data,
                **group_log_data,  
            }
            logger_data[model_name].append(detailed_log)
        
        if step >= total_train_steps:
            break
    

    
    if save_results:
        print("\n" + "=" * 80)
        print("Saving Results...")
        

        optimizer_names = "_vs_".join([str(cfg).replace("_", "-")[:20] for cfg in optimizer_configs])
        log_dir = os.path.join('logs', f'multi_training_{optimizer_names}_{str(uuid.uuid4())[:8]}')
        os.makedirs(log_dir, exist_ok=True)
        

        config_data = {
            'code': code,
            'config': config,
            'optimizer_configs': [str(cfg) for cfg in optimizer_configs],
        }
        

        for model_name in models.keys():
            if model_logs[model_name]:
                config_data[f'test_acc_{model_name}'] = model_logs[model_name][-1]['test_acc']
        
        torch.save(config_data, os.path.join(log_dir, 'config.pt'))
        

        for model_name in models.keys():
            safe_name = model_name.replace(".", "_").replace("/", "_")
            

            metrics_array = np.array([
                [log['epoch'], log['train_loss'], log['train_acc'], 
                 log['test_loss'], log['test_acc'], log['lr']]
                for log in model_logs[model_name]
            ])
            np.save(
                os.path.join(log_dir, f"metrics_{safe_name}.npy"),
                metrics_array,
                allow_pickle=True
            )
            

            with open(os.path.join(log_dir, f"metrics_{safe_name}.json"), 'w') as f:
                json.dump(model_logs[model_name], f, indent=2)
            

            with open(os.path.join(log_dir, f"logger_data_{safe_name}.pkl"), 'wb') as f:
                pickle.dump(logger_data[model_name], f)
        

        for model_name in models.keys():
            safe_name = model_name.replace(".", "_").replace("/", "_")
            with open(os.path.join(log_dir, f"singular_values_{safe_name}.pkl"), 'wb') as f:
                pickle.dump(singular_values_logs[model_name], f)
        

        for model_name, model in models.items():
            safe_name = model_name.replace(".", "_").replace("/", "_")
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_{safe_name}.pt"))
        

        readme_content = f"""# Multi-Model Training Experiment

## Configuration
- Architecture: {arch.__class__.__name__}
- Total Parameters: {sum(p.numel() for p in base_model.parameters()):,}
- Batch Size: {batch_size}
- Total Steps: {total_train_steps}
- Total Epochs: {total_epochs}

## Optimizers
"""
        for i, (opt_config, model_name) in enumerate(zip(optimizer_configs, models.keys())):
            final_test_acc = model_logs[model_name][-1]['test_acc'] if model_logs[model_name] else 0.0
            readme_content += f"{i+1}. {model_name}\n"
            readme_content += f"   - Final Test Accuracy: {final_test_acc:.4f}\n"
        
        with open(os.path.join(log_dir, "README.md"), 'w') as f:
            f.write(readme_content)

        print(f"  - Results saved to: {os.path.abspath(log_dir)}")
        print(f"  - Config: config.pt, README.md")
        print(f"  - Metrics: metrics_*.npy and metrics_*.json for each model")
        print(f"  - Detailed logger data: logger_data_*.pkl for each model (includes singular values, grad norms, etc.)")
        print(f"  - Singular values: singular_values_*.pkl for each model (legacy format)")
        print(f"  - Models: model_*.pt for each model")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    

    for model_name in models.keys():
        if model_logs[model_name]:
            final_test_acc = model_logs[model_name][-1]['test_acc']
            print(f"Final Test Accuracy ({model_name[:30]}): {final_test_acc:.4f}")
    
    print("=" * 80)
    

    results = {
        'model_logs': model_logs,
        'singular_values': singular_values_logs,
    }
    for model_name in models.keys():
        if model_logs[model_name]:
            results[f'test_acc_{model_name}'] = model_logs[model_name][-1]['test_acc']
    
    return results


if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])

