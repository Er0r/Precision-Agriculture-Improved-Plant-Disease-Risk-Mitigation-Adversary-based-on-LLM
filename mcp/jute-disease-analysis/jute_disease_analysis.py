#!/usr/bin/env python3
"""
Jute Disease Detection - Smart Pipeline

This script implements a smart jute disease detection system that:
- Checks for existing trained models first
- Skips training if model exists
- Goes directly to disease analysis

Detects 5 jute disease categories:
- Dieback
- Fresh (Healthy)
- Holed
- Mosaic
- Stem Soft Rot
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

# Enhanced Configuration for Better Accuracy
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch size for better gradient updates
EPOCHS = 35      # More epochs with early stopping
LEARNING_RATE = 0.0001  # Lower learning rate for transfer learning

# Dataset path
DATASET_PATH = "Dataset/Jute Disease Dataset/Jute Diesease Dataset"

# Model paths
MODEL_PATH = 'models/jute_disease_cnn_model.h5'
CONFIG_PATH = 'models/jute_disease_cnn_model_config.json'

# Class names
CLASS_NAMES = ['Dieback', 'Fresh', 'Holed', 'Mosaic', 'Stem Soft Rot']

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
    
    def create_improved_cnn_model(input_shape=(224, 224, 3), num_classes=5):
        """
        Improved CNN model with better architecture for higher accuracy
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers with regularization
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=5):
        """
        Transfer learning model using MobileNetV2 for better accuracy
        """
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    # Create improved model with transfer learning
    print("🚀 Creating improved model with transfer learning...")
    model, base_model = create_transfer_learning_model(input_shape=(*IMG_SIZE, 3), num_classes=len(CLASS_NAMES))
    
    # Compile with lower learning rate for transfer learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ New model created and compiled.")
    model.summary()
    
    # Set flag to proceed with training
    SKIP_TRAINING = False

# Data Loading (Only if training needed)
if not SKIP_TRAINING:
    print("📁 Setting up data generators for training...")
    
    # Enhanced data augmentation for better generalization
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data generator (only rescaling)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
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

    # Validation data (using separate generator without augmentation)
    validation_generator = val_datagen.flow_from_directory(
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
    print(f"🚀 Starting improved training process...")
    
    # Enhanced callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Phase 1: Train with frozen base model
    print("📚 Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning with unfrozen layers
    print("🔧 Phase 2: Fine-tuning with unfrozen layers...")
    base_model.trainable = True
    
    # Freeze early layers, unfreeze later layers for fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history2 = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = history1
    for key in history2.history:
        history.history[key].extend(history2.history[key])

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

    # Save enhanced model configuration
    model_config = {
        'model_name': 'jute_disease_improved_cnn_model',
        'model_type': 'transfer_learning_mobilenetv2',
        'input_shape': list(IMG_SIZE) + [3],
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['loss']),
        'final_accuracy': float(val_accuracy),
        'final_loss': float(val_loss),
        'training_phases': 2,
        'data_augmentation': 'enhanced',
        'architecture': 'MobileNetV2 + Custom Head'
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"📋 Model configuration saved to: {CONFIG_PATH}")
    
else:
    print("⏭️ SKIPPING TRAINING - Using existing trained model")
    print("💡 To retrain: delete model files and run again")

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

def analyze_jute_disease(image_path):
    """
    🎯 MAIN FUNCTION: Quick disease analysis for any jute leaf image
    
    Usage: 
    result = analyze_jute_disease('path/to/your/image.jpg')
    
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
            plt.title(f"JUTE DISEASE DIAGNOSIS\n{result['predicted_class']} (Confidence: {result['confidence']*100:.1f}%)", 
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
print("🔍 JUTE DISEASE ANALYSIS")
print("=" * 50)
print("Testing with sample images...")

sample_images = [
    'sample_images/fresh_sample.jpg',
    'sample_images/dieback_sample.jpg',
    'sample_images/holed_sample.jpg',
    'sample_images/mosaic_sample.jpg',
    'sample_images/stem_soft_rot_sample.jpg'
]

for image_path in sample_images:
    if os.path.exists(image_path):
        print(f"\n📸 Analyzing: {os.path.basename(image_path)}")
        print("-" * 30)
        
        result = predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
        
        if result:
            print(f"🎯 PREDICTION: {result['predicted_class']}")
            print(f"📊 CONFIDENCE: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            
            # Show confidence level
            if result['confidence'] > 0.8:
                print("✅ HIGH CONFIDENCE")
            elif result['confidence'] > 0.6:
                print("⚠️ MEDIUM CONFIDENCE")
            else:
                print("❌ LOW CONFIDENCE")
            
            print("\n📈 All probabilities:")
            for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)  # Simple progress bar
                print(f"   {class_name:20}: {prob:.3f} {bar}")
        else:
            print("❌ Failed to analyze image")
    else:
        print(f"❌ Sample image not found: {image_path}")
        print("   Make sure sample images are in the sample_images folder")

print("\n✅ DISEASE ANALYSIS COMPLETE!")

# System Summary
print("🎉 JUTE DISEASE DETECTION SYSTEM READY!")
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
print(f"   🔍 Detects {len(CLASS_NAMES)} jute disease categories")
print(f"   📐 Input image size: {IMG_SIZE}")
print(f"   🏷️ Classes: {', '.join(CLASS_NAMES)}")

print(f"\n🔧 MAIN FUNCTIONS:")
print(f"   🎯 analyze_jute_disease(image_path) - Main analysis function")
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
print(f"   Use: analyze_jute_disease('your_image.jpg')")

print("\n🚀 QUICK ANALYSIS FUNCTION READY!")
print("=" * 40)
print("Usage: analyze_jute_disease('path/to/your/image.jpg')")
print("\nExample:")
print("result = analyze_jute_disease('sample_images/fresh_sample.jpg')")
print("\n💡 This function works with any jute leaf image!")