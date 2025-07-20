"""
Rice Disease Detection Inference Script
Simple script for making predictions on new images
"""

import sys
import os
from pathlib import Path

# Add handler to path
sys.path.append('../handler')
from rice_disease_detector import RiceDiseaseDetector, quick_prediction

import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Rice Disease Detection Inference')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', help='Path to model file (optional)')
    parser.add_argument('--show', '-s', action='store_true', help='Show the input image')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    print(f"Analyzing image: {args.image}")
    
    # Make prediction
    result = quick_prediction(args.image, args.model)
    
    if result:
        print("\n=== Prediction Results ===")
        print(f"Predicted Disease: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Timestamp: {result['timestamp']}")
        
        if 'all_probabilities' in result:
            print("\nAll Class Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Show image if requested
        if args.show:
            try:
                img = Image.open(args.image)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.title(f"Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Could not display image: {e}")
    
    else:
        print("Prediction failed. Make sure the model is trained and saved.")

if __name__ == "__main__":
    main()