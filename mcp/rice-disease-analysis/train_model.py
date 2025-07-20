"""
Rice Disease Detection Training Script
Simple script to train the model from command line
"""

import sys
import os
from pathlib import Path

# Add handler to path
sys.path.append('../handler')
from rice_disease_detector import RiceDiseaseDetector

import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Rice Disease Detection Model')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--transfer-learning', '-t', action='store_true', help='Use transfer learning')
    parser.add_argument('--dataset-path', '-d', help='Path to dataset directory')
    parser.add_argument('--model-name', '-n', default='rice_disease_model', help='Model name for saving')
    
    args = parser.parse_args()
    
    print("=== Rice Disease Detection Training ===")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Transfer learning: {args.transfer_learning}")
    print(f"Model name: {args.model_name}")
    
    # Create detector
    if args.dataset_path:
        detector = RiceDiseaseDetector(dataset_path=args.dataset_path)
    else:
        detector = RiceDiseaseDetector()
    
    detector.batch_size = args.batch_size
    
    try:
        # Run complete pipeline
        print("\nStarting training pipeline...")
        results = detector.run_complete_pipeline(
            epochs=args.epochs, 
            use_transfer_learning=args.transfer_learning
        )
        
        # Save with custom name
        if args.model_name != 'rice_disease_model':
            detector.save_model(args.model_name)
        
        print(f"\n=== Training Complete ===")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"Model saved as: {args.model_name}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())