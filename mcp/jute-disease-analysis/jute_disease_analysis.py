#!/usr/bin/env python3
"""
Jute Disease Detection System

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

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.0001

# Dataset path
DATASET_PATH = "Dataset/Jute Disease Dataset/Jute Diesease Dataset"

MODEL_PATH = 'models/jute_disease_improved_model.h5'
CONFIG_PATH = 'models/jute_disease_improved_model_config.json'

CLASS_NAMES = ['Dieback', 'Fresh', 'Holed', 'Mosaic', 'Stem Soft Rot']

print(f"Dataset path: {DATASET_PATH}")
print(f"Classes: {CLASS_NAMES}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    print("🎯 Found existing model!")
    print("✅ Loading existing model...")
    
    model = keras.models.load_model(MODEL_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    
    print(f"✅ Model loaded successfully!")
    print(f"🏗️ Architecture: {model_config.get('architecture', 'Enhanced CNN')}")
    accuracy = model_config.get('final_accuracy', 'N/A')
    if isinstance(accuracy, (int, float)):
        print(f"📊 Previous training accuracy: {accuracy:.4f}")
    else:
        print(f"📊 Previous training accuracy: {accuracy}")
    print(f"🔄 Epochs trained: {model_config.get('epochs_trained', 'N/A')}")
    
    SKIP_TRAINING = True
    
else:
    print("❌ No existing model found")
    print("🔧 Will create and train model...")
    
    def create_improved_cnn_model(input_shape=(224, 224, 3), num_classes=5):
        """CNN model with enhanced architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.GlobalAveragePooling2D(),
            
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
        """Transfer learning model using MobileNetV2"""
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
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
    
    print("🚀 Creating model with transfer learning...")
    model, base_model = create_transfer_learning_model(input_shape=(*IMG_SIZE, 3), num_classes=len(CLASS_NAMES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created and compiled.")
    print("🏗️ Architecture: MobileNetV2 + Custom Head")
    model.summary()
    
    SKIP_TRAINING = False

if not SKIP_TRAINING:
    print("📁 Setting up data generators for training...")
    
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
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print("✅ Data generators created successfully!")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {validation_generator.samples}")
else:
    print("⏭️ Skipping data preparation - using existing trained model")

if not SKIP_TRAINING:
    print(f"🚀 Starting training process...")
    
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

    print("📚 Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("🔧 Phase 2: Fine-tuning with unfrozen layers...")
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    history = history1
    for key in history2.history:
        history.history[key].extend(history2.history[key])

    print("✅ Training completed!")
    
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    plot_training_history(history)
    
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"📊 Final validation accuracy: {val_accuracy:.4f}")
    
    os.makedirs('models', exist_ok=True)
    
    model.save(MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

    model_config = {
        'model_name': 'jute_disease_improved_model',
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
    print("⏭️ Skipping training - using existing model")
    print("💡 To retrain: delete model files and run again")

def predict_disease(image_path, model, class_names, img_size=(224, 224)):
    """Predict disease from a single image"""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
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
    Main function: Disease analysis for jute leaf images
    
    Usage: 
    result = analyze_jute_disease('path/to/your/image.jpg')
    
    Returns: Dictionary with prediction results
    """
    print(f"🔍 Analyzing: {os.path.basename(image_path)}")
    print("=" * 50)
    
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        result = predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
        
        if result:
            print(f"🎯 Diagnosis: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            
            if result['confidence'] > 0.9:
                print("🌟 Very high confidence")
            elif result['confidence'] > 0.8:
                print("✅ High confidence")
            elif result['confidence'] > 0.6:
                print("⚠️ Medium confidence")
            else:
                print("❌ Low confidence")
            
            print(f"\n📊 Top 3 predictions:")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"   {emoji} {class_name}: {prob:.4f} ({prob*100:.1f}%)")
            
            img = Image.open(image_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Jute Disease Diagnosis\n{result['predicted_class']} (Confidence: {result['confidence']*100:.1f}%)", 
                     fontsize=20, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            print("\n✅ Analysis complete!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return None

print("🔍 Jute Disease Analysis")
print("=" * 60)
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
        print("-" * 40)
        
        result = predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
        
        if result:
            print(f"🎯 Prediction: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            
            if result['confidence'] > 0.9:
                print("🌟 Very high confidence")
            elif result['confidence'] > 0.8:
                print("✅ High confidence")
            elif result['confidence'] > 0.6:
                print("⚠️ Medium confidence")
            else:
                print("❌ Low confidence")
            
            print("\n📈 All probabilities:")
            for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 25)
                print(f"   {class_name:20}: {prob:.3f} {bar}")
        else:
            print("❌ Failed to analyze image")
    else:
        print(f"❌ Sample image not found: {image_path}")
        print("   Make sure sample images are in the sample_images folder")

print("\n✅ Disease analysis complete!")

print("🎉 Jute Disease Detection System Ready!")
print("=" * 60)

if SKIP_TRAINING:
    print("✅ Used existing model")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        accuracy = config.get('final_accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            print(f"   📊 Model accuracy: {accuracy:.4f}")
        else:
            print(f"   📊 Model accuracy: {accuracy}")
        print(f"   🏗️ Architecture: {config.get('architecture', 'Enhanced CNN')}")
        print(f"   � Epochos trained: {config.get('epochs_trained', 'N/A')}")
else:
    print("✅ Successfully trained new model")

print(f"\n📋 System capabilities:")
print(f"   🔍 Detects {len(CLASS_NAMES)} jute disease categories")
print(f"   📐 Input image size: {IMG_SIZE}")
print(f"   🏷️ Classes: {', '.join(CLASS_NAMES)}")
print(f"   � Architec{ture: Transfer Learning (MobileNetV2)")

print(f"\n🔧 Main functions:")
print(f"   🎯 analyze_jute_disease(image_path)")
print(f"   📊 predict_disease(...)")

print(f"\n🔄 To retrain model:")
print(f"   1. Delete: {MODEL_PATH}")
print(f"   2. Delete: {CONFIG_PATH}")
print(f"   3. Run this script again")

print(f"\n🎯 Ready for disease analysis!")
print(f"   Use: analyze_jute_disease('your_image.jpg')")

print("\n🚀 Analysis function ready!")
print("=" * 50)
print("Usage: analyze_jute_disease('path/to/your/image.jpg')")
print("\nExample:")
print("result = analyze_jute_disease('sample_images/fresh_sample.jpg')")
print("\n💡 This function works with any jute leaf image!")