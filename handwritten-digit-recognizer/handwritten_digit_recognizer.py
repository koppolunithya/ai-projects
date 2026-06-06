import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HandwrittenDigitRecognizer:
    def __init__(self, model_name='mnist_cnn_model.h5'):
        """
        Initialize the Handwritten Digit Recognizer
        
        Args:
            model_name: Name to save the trained model
        """
        self.model_name = model_name
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load MNIST dataset and preprocess"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to (num_samples, 28, 28, 1) for CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Testing data shape: {x_test.shape}")
        print("Data preprocessing completed!")
        
    def build_model(self):
        """Build Convolutional Neural Network model"""
        print("\nBuilding CNN model...")
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
    def train(self, epochs=20, batch_size=128, validation_split=0.1):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        print("\nTraining model...")
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate(self):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes))
        
        return y_test_classes, y_pred_classes
    
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.save(self.model_name)
        print(f"Model saved as '{self.model_name}'")
    
    def load_model(self):
        """Load a previously trained model"""
        self.model = keras.models.load_model(self.model_name)
        print(f"Model loaded from '{self.model_name}'")
    
    def predict_single(self, image):
        """
        Predict digit for a single image
        
        Args:
            image: Input image (28x28 numpy array)
            
        Returns:
            Predicted digit and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        
        # Preprocess image
        image = image.astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("Model not trained.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved as 'training_history.png'")
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=True)
        plt.title('Confusion Matrix - MNIST Digit Recognition', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix plot saved as 'confusion_matrix.png'")
        plt.show()
    
    def plot_sample_predictions(self, num_samples=10):
        """Plot sample predictions"""
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for idx, ax in enumerate(axes):
            image = self.x_test[indices[idx]].reshape(28, 28)
            prediction = self.model.predict(self.x_test[indices[idx]:indices[idx]+1], verbose=0)
            predicted_digit = np.argmax(prediction)
            true_digit = np.argmax(self.y_test[indices[idx]])
            confidence = np.max(prediction)
            
            ax.imshow(image, cmap='gray')
            color = 'green' if predicted_digit == true_digit else 'red'
            ax.set_title(f'True: {true_digit}, Pred: {predicted_digit}\nConf: {confidence:.2f}', color=color)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        print("Sample predictions plot saved as 'sample_predictions.png'")
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = HandwrittenDigitRecognizer()
    
    # Load and preprocess data
    recognizer.load_and_preprocess_data()
    
    # Build model
    recognizer.build_model()
    
    # Train model
    recognizer.train(epochs=20, batch_size=128, validation_split=0.1)
    
    # Evaluate
    y_true, y_pred = recognizer.evaluate()
    
    # Save model
    recognizer.save_model()
    
    # Plot results
    recognizer.plot_training_history()
    recognizer.plot_confusion_matrix(y_true, y_pred)
    recognizer.plot_sample_predictions(num_samples=10)
    
    print("\nHandwritten Digit Recognition - All tasks completed!")