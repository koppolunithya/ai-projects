# AI Projects Repository

A comprehensive collection of AI and Machine Learning projects demonstrating various techniques and applications.

## 📚 Projects Included

### 1. Stock Price Predictor
**Location:** `stock-price-predictor/`  
**Branch:** `stock-price-predictor`

A machine learning model that predicts stock prices based on historical data using Linear Regression.

**Key Features:**
- Downloads historical stock data from Yahoo Finance
- Uses 60-day lookback window for feature engineering
- Trains Linear Regression model with performance metrics
- Predicts future stock prices (30 days by default)
- Generates comprehensive visualization plots

**Quick Start:**
```bash
cd stock-price-predictor
pip install -r requirements.txt
python stock_price_predictor.py
```

**Usage Example:**
```python
from stock_price_predictor import StockPricePredictor

predictor = StockPricePredictor(ticker='AAPL')
predictor.fetch_data()
results = predictor.train(lookback=60, test_size=0.2)
future_predictions = predictor.predict_future(days=30)
```

---

### 2. Handwritten Digit Recognizer
**Location:** `handwritten-digit-recognizer/`  
**Branch:** `handwritten-digit-recognizer`

A Convolutional Neural Network (CNN) model that recognizes handwritten digits from the MNIST dataset with >99% accuracy.

**Key Features:**
- 3 Convolutional blocks with batch normalization
- Dropout regularization for better generalization
- MNIST dataset (70,000 handwritten digits)
- Comprehensive evaluation metrics
- Generates confusion matrix and training plots
- Model persistence (save/load)

**Quick Start:**
```bash
cd handwritten-digit-recognizer
pip install -r requirements.txt
python handwritten_digit_recognizer.py
```

**Usage Example:**
```python
from handwritten_digit_recognizer import HandwrittenDigitRecognizer

recognizer = HandwrittenDigitRecognizer()
recognizer.load_and_preprocess_data()
recognizer.build_model()
recognizer.train(epochs=20, batch_size=128, validation_split=0.1)
y_true, y_pred = recognizer.evaluate()
```

---

## 📋 Repository Structure

```
ai-projects/
├── stock-price-predictor/
│   ├── stock_price_predictor.py
│   ├── README.md
│   └── requirements.txt
│
├── handwritten-digit-recognizer/
│   ├── handwritten_digit_recognizer.py
│   ├── README.md
│   └── requirements.txt
│
└── README.md (this file)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Virtual environment (recommended)

### Global Installation

Install all dependencies for both projects:
```bash
# Stock Price Predictor
cd stock-price-predictor
pip install -r requirements.txt

# Handwritten Digit Recognizer
cd ../handwritten-digit-recognizer
pip install -r requirements.txt
```

---

## 📊 Project Comparison

| Feature | Stock Price Predictor | Digit Recognizer |
|---------|----------------------|------------------|
| **Algorithm** | Linear Regression | CNN (Deep Learning) |
| **Data Source** | Yahoo Finance | MNIST Dataset |
| **Input** | Historical stock prices | 28x28 digit images |
| **Output** | Price predictions | Digit classification |
| **Accuracy** | RMSE-based | >99% accuracy |
| **Training Time** | Seconds | ~5-10 minutes |
| **Use Case** | Financial forecasting | Image classification |

---

## 🔧 Technologies Used

### Stock Price Predictor
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **scikit-learn**: Machine learning (Linear Regression)
- **yfinance**: Stock data fetching
- **Matplotlib**: Data visualization

### Handwritten Digit Recognizer
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **scikit-learn**: Metrics & evaluation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualization

---

## 📈 Performance Metrics

### Stock Price Predictor
- **Training RMSE**: ~0.02-0.05 (normalized scale)
- **Testing RMSE**: ~0.03-0.06 (normalized scale)
- **R² Score**: ~0.85-0.95
- **Prediction Type**: Time-series regression

### Handwritten Digit Recognizer
- **Training Accuracy**: >99%
- **Testing Accuracy**: >99%
- **Per-class F1-Score**: >0.98
- **Prediction Type**: Multi-class classification

---

## 💾 Output Files

### Stock Price Predictor
- `stock_price_predictions.png`: Training and prediction visualization

### Handwritten Digit Recognizer
- `mnist_cnn_model.h5`: Trained model weights
- `training_history.png`: Accuracy and loss curves
- `confusion_matrix.png`: Per-digit prediction accuracy
- `sample_predictions.png`: Sample predictions with confidence

---

## 🚀 Usage Examples

### Example 1: Predict Apple Stock Prices
```python
from stock_price_predictor import StockPricePredictor

# Create predictor for Apple stock
predictor = StockPricePredictor(ticker='AAPL')

# Fetch data from last 2 years
predictor.fetch_data()

# Train model
X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = predictor.train()

# Predict next 30 days
predictions = predictor.predict_future(days=30)

# Display predictions
for i, price in enumerate(predictions[:5], 1):
    print(f"Day {i}: ${price:.2f}")

# Plot results
predictor.plot_results(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, predictions)
```

### Example 2: Recognize Handwritten Digits
```python
from handwritten_digit_recognizer import HandwrittenDigitRecognizer
import numpy as np

# Initialize recognizer
recognizer = HandwrittenDigitRecognizer()

# Load and preprocess MNIST data
recognizer.load_and_preprocess_data()

# Build and train model
recognizer.build_model()
recognizer.train(epochs=20)

# Evaluate model
y_true, y_pred = recognizer.evaluate()

# Visualize results
recognizer.plot_training_history()
recognizer.plot_confusion_matrix(y_true, y_pred)
recognizer.plot_sample_predictions(num_samples=10)

# Predict single digit
test_image = recognizer.x_test[0]
digit, confidence = recognizer.predict_single(test_image)
print(f"Predicted: {digit}, Confidence: {confidence:.2f}")
```

---

## 📝 Notes & Limitations

### Stock Price Predictor
- ⚠️ **Not for trading decisions**: Use for educational purposes only
- Linear Regression may not capture complex market patterns
- Stock prices depend on many factors not in the model
- Past performance doesn't guarantee future results

### Handwritten Digit Recognizer
- ⚠️ Trained only on MNIST dataset (0-9 digits)
- May not generalize well to handwriting styles very different from MNIST
- Requires input images to be 28x28 pixels
- Includes dropout and batch normalization for regularization

---

## 🔮 Future Enhancements

### Stock Price Predictor
- [ ] Implement LSTM/RNN for better time-series modeling
- [ ] Add technical indicators (moving averages, RSI, MACD)
- [ ] Ensemble methods for improved predictions
- [ ] Error bars and confidence intervals
- [ ] Multi-step ahead predictions

### Handwritten Digit Recognizer
- [ ] Data augmentation (rotation, scaling, shifting)
- [ ] Transfer learning with pre-trained models
- [ ] Ensemble of multiple models
- [ ] Web service deployment
- [ ] Custom dataset fine-tuning
- [ ] Attention mechanisms for interpretability

---

## 📚 References & Resources

### Stock Price Prediction
- [Yahoo Finance API](https://finance.yahoo.com/)
- [Time Series Forecasting Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Deep Learning & MNIST
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition"

---

## 🤝 Contributing

Feel free to fork, modify, and improve these projects. Suggestions for enhancements:
- Add new models or algorithms
- Improve documentation
- Add unit tests
- Create Jupyter notebooks with detailed explanations
- Add new projects to the repository

---

## 📄 License

These projects are provided for educational purposes.

---

## 👨‍💻 Author

**koppolunithya**

Created: June 2026

---

## ⭐ Project Status

✅ **Stock Price Predictor**: Complete  
✅ **Handwritten Digit Recognizer**: Complete  

---

## 📞 Support

For questions or issues:
1. Check the individual README files in each project folder
2. Review the code comments for detailed explanations
3. Refer to the documentation links provided above

---

## 🎯 Learning Outcomes

By working through these projects, you'll learn:

1. **Stock Price Predictor:**
   - Time-series data handling
   - Feature engineering techniques
   - Model evaluation metrics (RMSE, R²)
   - Data visualization
   - Real-world API integration

2. **Handwritten Digit Recognizer:**
   - Convolutional Neural Networks (CNNs)
   - Deep learning frameworks (TensorFlow/Keras)
   - Image preprocessing and normalization
   - Model training and evaluation
   - Classification metrics and confusion matrices
   - Regularization techniques

---

**Last Updated:** June 6, 2026
