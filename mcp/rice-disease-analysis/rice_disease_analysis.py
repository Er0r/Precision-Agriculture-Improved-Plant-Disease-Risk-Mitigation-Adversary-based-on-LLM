#!/usr/bin/env python3
"""
Rice Disease Detection - Smart Pipeline

This script implements a smart rice disease detection system that:
- Checks for existing trained models first
- Skips training if model exists
- Goes directly to disease analysis

Detects 5 rice disease categories:
- Bacterial Leaf Blight
- Healthy Leaf  
- Rice
- Rice Blast
- Tungro
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from PIL import Image
import json

print("✅ Setup complete!")
print(f"TensorFlow version: {tf.__version__}")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Dataset path
DATASET_PATH = "dataset/Rice Leaf and Crop Disease Detection Dataset"

# Model paths
MODEL_PATH = Path(__file__).parent / "models" / "rice_disease_cnn_model.h5"
CONFIG_PATH = Path(__file__).parent / "models" / "rice_disease_cnn_model_config.json"

# Class names
CLASS_NAMES = ['Bacterial Leaf Blight', 'Healthy _leaf', 'Rice', 'Rice Blast', 'Tungro']

print(f"Dataset path: {DATASET_PATH}")
print(f"Classes: {CLASS_NAMES}")
print(f"Image size: {IMG_SIZE}")

# Smart Model Loading - Check for Existing Model
if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    print("🎯 FOUND EXISTING TRAINED MODEL!")
    print("=" * 40)
    print("✅ Loading existing model...")
    
    model = keras.models.load_model(MODEL_PATH)
    
    # Load model configuration
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    
    print(f"✅ Model loaded successfully!")
    # Safe formatting for accuracy
    accuracy = model_config.get('final_accuracy', 'N/A')
    if isinstance(accuracy, (int, float)):
        print(f"📊 Previous training accuracy: {accuracy:.4f}")
    else:
        print(f"📊 Previous training accuracy: {accuracy}")
    print(f"🔄 Epochs trained: {model_config.get('epochs_trained', 'N/A')}")
    print(f"⏰ Skipping training - going directly to disease analysis!")
    
    # Set flag to skip training
    SKIP_TRAINING = True
    
else:
    print("❌ NO EXISTING MODEL FOUND")
    print("=" * 30)
    print("🔧 Will create and train new model...")
    
    def create_cnn_model(input_shape=(224, 224, 3), num_classes=5):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    # Create and compile new model
    model = create_cnn_model(input_shape=(*IMG_SIZE, 3), num_classes=len(CLASS_NAMES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ New model created and compiled.")
    model.summary()
    
    # Set flag to proceed with training
    SKIP_TRAINING = False

# Only run training code when script is executed directly
if __name__ == "__main__":
    # Data Loading (Only if training needed)
    if not SKIP_TRAINING:
        print("📁 Setting up data generators for training...")
        
        # Create data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        # Training data
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        # Validation data
        validation_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        print("✅ Data generators created successfully!")
    else:
        print("⏭️ SKIPPING data preparation - using existing trained model")

    # Training (Only if needed)
    if not SKIP_TRAINING:
        print(f"🚀 Starting training for {EPOCHS} epochs...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        ]

        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        print("✅ Training completed!")
        
        # Plot training history
        def plot_training_history(history):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

        plot_training_history(history)
        
        # Evaluate and save model
        val_loss, val_accuracy = model.evaluate(validation_generator)
        print(f"📊 Final validation accuracy: {val_accuracy:.4f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model.save(MODEL_PATH)
        print(f"💾 Model saved to: {MODEL_PATH}")

        # Save model configuration
        model_config = {
            'model_name': 'rice_disease_cnn_model',
            'input_shape': list(IMG_SIZE) + [3],
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES,
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history.history['loss']),
            'final_accuracy': float(val_accuracy),
            'final_loss': float(val_loss)
        }

        with open(CONFIG_PATH, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"💾 Model configuration saved to: {CONFIG_PATH}")
        
        print("🎯 Training and saving complete!")
else:
    # When imported as a module, just ensure model is loaded
    if 'model' in globals() and model is not None:
        print("✅ Rice disease model is loaded and ready for inference!")
    else:
        print("⚠️ Rice disease model not loaded yet - will be loaded on first use")

# DISEASE ANALYSIS - The Main Feature!
def predict_disease(image_path, model, class_names, img_size=(224, 224)):
    """
    Predict disease from a single image
    """
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get all probabilities
        all_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_probabilities': all_probabilities
        }
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

def analyze_rice_disease(image_path):
    """
    🎯 MAIN FUNCTION: Quick disease analysis for any rice leaf image
    
    Usage: 
    result = analyze_rice_disease('path/to/your/image.jpg')
    
    Returns: Dictionary with prediction results
    """
    print(f"🔍 ANALYZING: {os.path.basename(image_path)}")
    print("=" * 40)
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        # Make prediction
        result = predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
        
        if result:
            print(f"🎯 DIAGNOSIS: {result['predicted_class']}")
            print(f"📊 CONFIDENCE: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            
            # Confidence assessment
            if result['confidence'] > 0.8:
                print("✅ HIGH CONFIDENCE - Reliable diagnosis")
            elif result['confidence'] > 0.6:
                print("⚠️ MEDIUM CONFIDENCE - Consider additional analysis")
            else:
                print("❌ LOW CONFIDENCE - Image may be unclear or unusual")
            
            # Show top 3 predictions
            print(f"\n📊 TOP 3 PREDICTIONS:")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"   {emoji} {class_name}: {prob:.4f} ({prob*100:.1f}%)")
            
            # Display image with result
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"RICE DISEASE DIAGNOSIS\n{result['predicted_class']} (Confidence: {result['confidence']*100:.1f}%)", 
                     fontsize=18, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            print("\n✅ Analysis complete!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return None

# Test with sample images
print("🔍 RICE DISEASE ANALYSIS")
print("=" * 50)
print("Testing with sample images...")

# Only run this when the script is executed directly, not when imported
if __name__ == "__main__":
    sample_images = [
        'sample_images/healthy_leaf_sample.jpg',
        'sample_images/rice_blast_sample.jpg'
    ]

    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"\n📸 Analyzing: {os.path.basename(image_path)}")
            print("-" * 30)
            analyze_rice_disease(image_path)
        else:
            print(f"❌ Sample image not found: {image_path}")
    
    print("\n🎯 Testing complete!")
else:
    # When imported as a module, just print basic info
    print("🌾 Rice Disease Analysis module loaded successfully!")
    print(f"📊 Available classes: {CLASS_NAMES}")
    print(f"🖼️ Image size: {IMG_SIZE}")
    if 'model' in globals():
        print("✅ Model is loaded and ready for inference!")
    else:
        print("⚠️ Model not loaded yet - will be loaded on first use")

# System Summary
print("🎉 RICE DISEASE DETECTION SYSTEM READY!")
print("=" * 50)

if SKIP_TRAINING:
    print("✅ SMART LOADING: Used existing trained model")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        # Safe formatting for accuracy
        accuracy = config.get('final_accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            print(f"   📊 Model accuracy: {accuracy:.4f}")
        else:
            print(f"   📊 Model accuracy: {accuracy}")
        print(f"   🔄 Epochs trained: {config.get('epochs_trained', 'N/A')}")
        print(f"   ⚡ Time saved: No training needed!")
else:
    print("✅ NEW MODEL: Successfully trained new model")

print(f"\n📋 SYSTEM CAPABILITIES:")
print(f"   🔍 Detects {len(CLASS_NAMES)} rice disease categories")
print(f"   📐 Input image size: {IMG_SIZE}")
print(f"   🏷️ Classes: {', '.join(CLASS_NAMES)}")

print(f"\n🔧 MAIN FUNCTIONS:")
print(f"   🎯 analyze_rice_disease(image_path) - Main analysis function")
print(f"   📊 predict_disease(...) - Detailed prediction function")

print(f"\n💡 SMART FEATURES:")
print(f"   ✅ Auto-detects existing models")
print(f"   ⏭️ Skips training if model exists")
print(f"   🚀 Goes directly to disease analysis")
print(f"   📊 Confidence-based recommendations")

print(f"\n🔄 TO RETRAIN MODEL:")
print(f"   1. Delete: {MODEL_PATH}")
print(f"   2. Delete: {CONFIG_PATH}")
print(f"   3. Run this script again")

print(f"\n🎯 READY FOR DISEASE ANALYSIS!")
print(f"   Use: analyze_rice_disease('your_image.jpg')")

print("\n🚀 QUICK ANALYSIS FUNCTION READY!")
print("=" * 40)
print("Usage: analyze_rice_disease('path/to/your/image.jpg')")
print("\nExample:")
print("result = analyze_rice_disease('sample_images/healthy_leaf_sample.jpg')")
print("\n💡 This function works with any rice leaf image!")