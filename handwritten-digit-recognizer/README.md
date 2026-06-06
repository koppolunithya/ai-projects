# Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) model that recognizes handwritten digits from the MNIST dataset with high accuracy.

## Features

- **Convolutional Neural Network**: Deep learning architecture with 3 convolutional blocks
- **Data Augmentation**: Batch normalization and dropout for regularization
- **MNIST Dataset**: 70,000 images of handwritten digits (0-9)
- **High Accuracy**: Achieves >99% accuracy on test set
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Confusion matrix and classification reports
- **Visualization**: Training history and sample predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train Model from Scratch

```python
from handwritten_digit_recognizer import HandwrittenDigitRecognizer

# Initialize recognizer
recognizer = HandwrittenDigitRecognizer()

# Load and preprocess MNIST data
recognizer.load_and_preprocess_data()

# Build CNN model
recognizer.build_model()

# Train model (20 epochs by default)
recognizer.train(epochs=20, batch_size=128, validation_split=0.1)

# Evaluate on test set
y_true, y_pred = recognizer.evaluate()

# Save model for future use
recognizer.save_model()

# Visualize results
recognizer.plot_training_history()
recognizer.plot_confusion_matrix(y_true, y_pred)
recognizer.plot_sample_predictions(num_samples=10)
```

### Use Pre-trained Model

```python
# Initialize and load existing model
recognizer = HandwrittenDigitRecognizer()
recognizer.load_and_preprocess_data()
recognizer.load_model()

# Make predictions
recognizer.plot_sample_predictions(num_samples=10)
```

### Predict Single Digit

```python
import numpy as np

# Create or load a 28x28 image
image = np.random.rand(28, 28) * 255  # Example image

# Predict
predicted_digit, confidence = recognizer.predict_single(image)
print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}")
```

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters) + BatchNorm + MaxPool + Dropout
    ↓
Conv2D (64 filters) + BatchNorm + MaxPool + Dropout
    ↓
Conv2D (128 filters) + BatchNorm + Dropout
    ↓
Flatten
    ↓
Dense (256) + BatchNorm + Dropout
    ↓
Dense (128) + BatchNorm + Dropout
    ↓
Dense (10, softmax) → Output
```

## Training Parameters

- **Epochs**: 20 (can be adjusted)
- **Batch Size**: 128
- **Validation Split**: 10%
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Regularization**: Batch Normalization, Dropout (0.25-0.5)

## Dataset Information

- **Total Images**: 70,000 (60,000 training + 10,000 testing)
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 10 digits (0-9)
- **Source**: MNIST (Modified National Institute of Standards and Technology)

## Expected Performance

- **Training Accuracy**: >99%
- **Test Accuracy**: >99%
- **Per-class F1-Score**: >0.98

## Output Files

- `mnist_cnn_model.h5`: Trained model weights
- `training_history.png`: Accuracy and loss curves
- `confusion_matrix.png`: Prediction accuracy per digit
- `sample_predictions.png`: Sample predictions with confidence scores

## Limitations

- Only trained on handwritten digits (0-9)
- May not generalize well to significantly different handwriting styles
- Requires images to be preprocessed to 28x28 pixels

## Future Improvements

- Implement data augmentation for better generalization
- Create ensemble of multiple models
- Deploy as web service for real-time predictions
- Fine-tune on custom handwritten digit datasets
- Implement attention mechanisms for better interpretability

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.