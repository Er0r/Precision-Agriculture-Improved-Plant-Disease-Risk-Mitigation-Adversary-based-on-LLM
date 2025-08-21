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
layers = keras.layers
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
from typing import Any, cast

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.0001

# Dataset path
DATASET_PATH = "Dataset/Jute Disease Dataset/Jute Diesease Dataset"

MODEL_PATH = 'models/jute_disease_cnn_model.h5'
CONFIG_PATH = 'models/jute_disease_cnn_model_config.json'

CLASS_NAMES = ['Dieback', 'Fresh', 'Holed', 'Mosaic', 'Stem Soft Rot']

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return model

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

def predict_disease(image_path, model_instance=None, class_names=None, img_size=(224, 224)):
    """Predict disease from a single image"""
    try:
        if model_instance is None:
            model_instance = load_model()
        if class_names is None:
            class_names = CLASS_NAMES
            
        img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Ensure model_instance is not None for both runtime and static checkers
        assert model_instance is not None, "Model instance is None; failed to load model"
        predictions = model_instance.predict(img_array)
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
    try:
        if not os.path.exists(image_path):
            return None
            
        model_instance = load_model()
        result = predict_disease(image_path, model_instance, CLASS_NAMES, IMG_SIZE)
        
        return result
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def train_model():
    """Train the jute disease detection model"""
    global model
    
    model, base_model = create_transfer_learning_model(input_shape=(*IMG_SIZE, 3), num_classes=len(CLASS_NAMES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
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
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
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
            min_lr=cast(Any, 1e-7),
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Phase 1: Training with frozen base model
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose='auto'
    )
    
    # Phase 2: Fine-tuning with unfrozen layers
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
        verbose='auto'
    )
    
    # Combine histories safely into a single History-like object
    combined_history = {}
    if history1 is not None and hasattr(history1, 'history'):
        for k, v in history1.history.items():
            combined_history[k] = list(v)
    if history2 is not None and hasattr(history2, 'history'):
        for k, v in history2.history.items():
            if k in combined_history:
                combined_history[k].extend(v)
            else:
                combined_history[k] = list(v)
    # Create a History object to hold the combined history so subsequent code can access history.history
    history = keras.callbacks.History()
    history.history = combined_history

    val_loss, val_accuracy = model.evaluate(validation_generator)
    
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)

    model_config = {
        'model_name': 'jute_disease_cnn_model',
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
    
    return model, history

if __name__ == "__main__":
    # Load model if exists, otherwise show training message
    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
        model = load_model()
        print("✅ Jute Disease Detection System Ready!")
        print(f"📋 System capabilities:")
        print(f"   🔍 Detects {len(CLASS_NAMES)} jute disease categories")
        print(f"   📐 Input image size: {IMG_SIZE}")
        print(f"   🏷️ Classes: {', '.join(CLASS_NAMES)}")
        
        # Test with sample images if available
        sample_images = [
            'sample_images/fresh_sample.jpg',
            'sample_images/dieback_sample.jpg', 
            'sample_images/holed_sample.jpg',
            'sample_images/mosaic_sample.jpg',
            'sample_images/stem_soft_rot_sample.jpg'
        ]
        
        print("\n🔍 Testing with sample images...")
        for image_path in sample_images:
            if os.path.exists(image_path):
                result = analyze_jute_disease(image_path)
                if result:
                    print(f"📸 {os.path.basename(image_path)}: {result['predicted_class']} ({result['confidence']:.2%})")
        
    else:
        print("❌ No trained model found.")
        print("Run train_model() to train a new model.")
