"""
Test script to verify the jute disease detection setup
"""

import sys
import os
from pathlib import Path

# Add handler to path
current_dir = Path(__file__).parent
handler_path = current_dir.parent / 'handler'
sys.path.insert(0, str(handler_path))

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing imports...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    return True

def test_detector_creation():
    """Test if the detector can be created"""
    print("\nTesting detector creation...")
    try:
        from jute_disease_detector import JuteDiseaseDetector, create_detector
        detector = create_detector()
        print(f"✓ Detector created successfully")
        print(f"  Dataset path: {detector.dataset_path}")
        print(f"  Output directory: {detector.output_dir}")
        print(f"  Classes: {detector.classes}")
        return True, detector
    except Exception as e:
        print(f"✗ Detector creation failed: {e}")
        return False, None

def test_dataset_access(detector):
    """Test if the dataset can be accessed"""
    print("\nTesting dataset access...")
    try:
        class_counts = detector.load_and_explore_dataset()
        
        total_original = sum(counts['original'] for counts in class_counts.values())
        total_augmented = sum(counts['augmented'] for counts in class_counts.values())
        
        if total_original > 0:
            print(f"✓ Found {total_original} original images across all classes")
            if total_augmented > 0:
                print(f"✓ Found {total_augmented} additional augmented images")
            return True
        else:
            print("⚠ Warning: No images found in dataset")
            return False
            
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False

def test_output_directories(detector):
    """Test if output directories are created"""
    print("\nTesting output directories...")
    try:
        required_dirs = [
            detector.output_dir,
            detector.output_dir / "models",
            detector.output_dir / "plots"
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✓ {dir_path} exists")
            else:
                print(f"✗ {dir_path} not found")
                return False
        
        print("✓ All output directories ready")
        return True
    except Exception as e:
        print(f"✗ Output directory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Jute Disease Detection Setup Test ===\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages:")
        print("pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn seaborn pillow")
        return False
    
    # Test 2: Detector creation
    success, detector = test_detector_creation()
    if not success:
        print("\n❌ Detector creation failed.")
        return False
    
    # Test 3: Dataset access
    if not test_dataset_access(detector):
        print("\n⚠ Dataset access issues detected.")
        print("Please ensure the dataset is located at:")
        print(f"  {detector.dataset_path}")
        print("And contains folders for each class:")
        for class_name in detector.classes:
            print(f"  - {class_name}/")
    
    # Test 4: Output directories
    if not test_output_directories(detector):
        print("\n❌ Output directory setup failed.")
        return False
    
    print("\n✅ Setup test completed successfully!")
    print("\nYou can now:")
    print("1. Run the Jupyter notebook: jute_disease_analysis.ipynb")
    print("2. Use the training script: python train_model.py")
    print("3. Use the inference script: python inference.py")
    
    return True

if __name__ == "__main__":
    main()