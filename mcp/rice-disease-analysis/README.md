# Rice Disease Detection System

A comprehensive CNN-based system for detecting rice leaf diseases using deep learning.

## Features

- **Multi-class Classification**: Detects 5 different rice conditions:
  - Bacterial Leaf Blight
  - Healthy Leaf
  - Rice
  - Rice Blast
  - Tungro

- **Data Augmentation**: Automatic image augmentation to improve model robustness
- **Transfer Learning**: Option to use pre-trained MobileNetV2 for better performance
- **Complete Pipeline**: From data loading to model deployment
- **Evaluation Tools**: Comprehensive model evaluation with confusion matrix and classification reports
- **Easy Inference**: Simple prediction interface for new images

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is located at:
```
F:\Personal\PrecisionAgriculture\dataset\Rice Leaf and Crop Disease Detection Dataset\
```

## Usage

### Option 1: Jupyter Notebook (Recommended for Development)

1. Open the Jupyter notebook:
```bash
jupyter notebook rice_disease_analysis.ipynb
```

2. Follow the step-by-step cells to:
   - Explore the dataset
   - Train the model
   - Evaluate performance
   - Make predictions

### Option 2: Python Script

```python
from rice_disease_detector import RiceDiseaseDetector

# Create detector
detector = RiceDiseaseDetector()

# Run complete pipeline
results = detector.run_complete_pipeline(epochs=30, use_transfer_learning=True)

# Make prediction on new image
result = detector.predict_disease("path/to/image.jpg", return_probabilities=True)
print(f"Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
```

### Option 3: Command Line Inference

After training, use the inference script:

```bash
python inference.py --image path/to/image.jpg --show
```

Options:
- `--image, -i`: Path to input image (required)
- `--model, -m`: Path to model file (optional)
- `--show, -s`: Display the input image
- `--output, -o`: Save results to JSON file

## Model Architecture

### Custom CNN
- 4 Convolutional layers with MaxPooling
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for multi-class output

### Transfer Learning (Recommended)
- Pre-trained MobileNetV2 base
- Custom classification head
- Fine-tuning capabilities

## Data Preprocessing

1. **Image Resizing**: All images resized to 224x224 pixels
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Data Augmentation**: 
   - Rotation (±20°)
   - Width/Height shift (±20%)
   - Shear transformation
   - Zoom (±20%)
   - Horizontal flip

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**:
  - Early Stopping (patience=10)
  - Learning Rate Reduction
  - Model Checkpointing

## Output Files

The system generates several output files:

- `models/`: Trained model files
- `plots/`: Visualization plots (training history, confusion matrix, class distribution)
- `training_history.csv`: Training metrics per epoch
- `evaluation_results.json`: Final evaluation metrics

## Performance Monitoring

The system provides comprehensive evaluation:

- **Training History Plots**: Accuracy and loss curves
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, recall, F1-score per class
- **Test Accuracy**: Final model performance on unseen data

## Example Results Structure

```json
{
  "predicted_class": "Rice Blast",
  "confidence": 0.8945,
  "timestamp": "2024-01-15T10:30:45",
  "all_probabilities": {
    "Bacterial Leaf Blight": 0.0234,
    "Healthy _leaf": 0.0456,
    "Rice": 0.0123,
    "Rice Blast": 0.8945,
    "Tungro": 0.0242
  }
}
```

## Tips for Best Results

1. **Use Transfer Learning**: Generally provides better accuracy with less training time
2. **Monitor Training**: Watch for overfitting using validation curves
3. **Data Quality**: Ensure images are clear and properly labeled
4. **Augmentation**: Helps with limited data and improves generalization
5. **Early Stopping**: Prevents overfitting and saves training time

## Troubleshooting

### Common Issues:

1. **Dataset Not Found**: Ensure the dataset path is correct
2. **Memory Issues**: Reduce batch size or image size
3. **Poor Performance**: Try transfer learning or increase training epochs
4. **Overfitting**: Increase dropout rates or use more data augmentation

### Performance Optimization:

- Use GPU if available for faster training
- Adjust batch size based on available memory
- Use mixed precision training for faster computation
- Consider data pipeline optimization for large datasets

## Future Enhancements

- [ ] Add image segmentation preprocessing
- [ ] Implement ensemble methods
- [ ] Add real-time camera capture
- [ ] Web interface for easy deployment
- [ ] Mobile app integration
- [ ] Additional disease classes
- [ ] Severity assessment
- [ ] Treatment recommendations