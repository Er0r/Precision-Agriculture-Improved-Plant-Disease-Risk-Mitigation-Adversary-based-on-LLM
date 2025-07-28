#!/usr/bin/env python3
"""
Test script to verify sample images work with the jute disease analysis system
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

def test_sample_images():
    """Test all sample images"""
    print("🧪 TESTING SAMPLE IMAGES")
    print("=" * 40)
    
    sample_images = [
        ('sample_images/fresh_sample.jpg', 'Fresh (Healthy)'),
        ('sample_images/dieback_sample.jpg', 'Dieback'),
        ('sample_images/holed_sample.jpg', 'Holed'),
        ('sample_images/mosaic_sample.jpg', 'Mosaic'),
        ('sample_images/stem_soft_rot_sample.jpg', 'Stem Soft Rot')
    ]
    
    for image_path, disease_name in sample_images:
        print(f"\n📸 Testing: {disease_name}")
        print("-" * 30)
        
        if os.path.exists(image_path):
            try:
                # Try to open and display basic info about the image
                img = Image.open(image_path)
                print(f"✅ Image loaded successfully")
                print(f"   📐 Size: {img.size}")
                print(f"   🎨 Mode: {img.mode}")
                print(f"   📁 File: {os.path.basename(image_path)}")
                
                # Display the image
                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.title(f"Sample: {disease_name}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"❌ Error loading image: {e}")
        else:
            print(f"❌ Image not found: {image_path}")
    
    print(f"\n✅ SAMPLE IMAGE TEST COMPLETE!")
    print(f"📋 Found {len([img for img, _ in sample_images if os.path.exists(img)])} out of {len(sample_images)} sample images")

def show_dataset_info():
    """Show information about the dataset"""
    print("\n📊 DATASET INFORMATION")
    print("=" * 40)
    
    dataset_path = "Dataset/Jute Disease Dataset/Jute Diesease Dataset"
    
    if os.path.exists(dataset_path):
        disease_folders = [
            'Dieback-300',
            'Fresh-280', 
            'Holed-300',
            'Mosaic-240',
            'Stem Soft Rot-270'
        ]
        
        total_images = 0
        for folder in disease_folders:
            folder_path = os.path.join(dataset_path, folder)
            if os.path.exists(folder_path):
                image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
                total_images += image_count
                print(f"📁 {folder}: {image_count} images")
            else:
                print(f"❌ {folder}: Folder not found")
        
        print(f"\n📈 Total images in dataset: {total_images}")
    else:
        print(f"❌ Dataset not found at: {dataset_path}")

if __name__ == "__main__":
    test_sample_images()
    show_dataset_info()
    
    print(f"\n🎯 READY FOR ANALYSIS!")
    print(f"Use the main script: python jute_disease_analysis.py")
    print(f"Or try: analyze_jute_disease('sample_images/fresh_sample.jpg')")