import os
import yaml

def create_config_files():
    """Create configuration files for EchoLip-Ai"""
    
    # Training configuration
    train_config = {
        'model': {
            'audio_encoder': 'conv1d_lstm',
            'face_encoder': 'resnet50',
            'fusion_type': 'cross_attention',
            'lip_generator': 'unet',
            'face_compositor': 'blending_network'
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.0001,
            'epochs': 100,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'gradient_clipping': 1.0
        },
        'data': {
            'train_path': 'data/datasets/train',
            'val_path': 'data/datasets/val',
            'test_path': 'data/datasets/test',
            'frame_size': [256, 256],
            'fps': 25,
            'audio_sample_rate': 16000
        },
        'losses': {
            'l1_weight': 1.0,
            'perceptual_weight': 0.1,
            'sync_weight': 0.5,
            'gan_weight': 0.01
        },
        'logging': {
            'experiment_name': 'echolip_v1',
            'log_dir': 'logs',
            'save_frequency': 10,
            'log_frequency': 100
        }
    }
    
    # Model configuration
    model_config = {
        'audio_encoder': {
            'input_dim': 80,
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout': 0.1
        },
        'face_encoder': {
            'backbone': 'resnet50',
            'pretrained': True,
            'feature_dim': 2048
        },
        'fusion_module': {
            'audio_dim': 512,
            'face_dim': 2048,
            'hidden_dim': 1024,
            'num_heads': 8
        },
        'lip_generator': {
            'input_channels': 3,
            'base_channels': 64,
            'num_layers': 4
        }
    }
    
    # Inference configuration
    inference_config = {
        'model_path': 'checkpoints/best_model.pth',
        'device': 'cuda',
        'batch_size': 1,
        'output_format': 'mp4',
        'quality': 'high',
        'real_time': True
    }
    
    # Create configs directory
    os.makedirs('configs', exist_ok=True)
    
    # Save configuration files
    configs = {
        'configs/train_config.yaml': train_config,
        'configs/model_config.yaml': model_config,
        'configs/inference_config.yaml': inference_config
    }
    
    for file_path, config in configs.items():
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration files created successfully!")

if __name__ == '__main__':
    create_config_files()