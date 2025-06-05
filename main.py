#!/usr/bin/env python3
"""EchoLip-Ai: Real-time Audio-Visual Lip Synchronization"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='EchoLip-Ai Lip Sync System')
    parser.add_argument('--mode', choices=['train', 'inference', 'evaluate'], 
                       required=True, help='Mode to run')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print(f"Running in {args.mode} mode...")

if __name__ == '__main__':
    main()
