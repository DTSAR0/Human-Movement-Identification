# Human Movement Identification

A project for Human Activity Recognition (HAR) using the UCI HAR Dataset. This project implements and compares two machine learning approaches: Random Forest and Multi-Layer Perceptron (MLP) neural networks for classifying human activities from smartphone sensor data.

## 📋 Project Overview

This project focuses on recognizing human activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) using accelerometer and gyroscope data from smartphones. The dataset contains 561 engineered features extracted from raw sensor signals.

### Dataset

- **UCI HAR Dataset**: Human Activity Recognition Using Smartphones Dataset
- **Features**: 561 engineered features (time and frequency domain features)
- **Classes**: 6 activities
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING
- **Samples**: ~10,299 total (7,352 train + 2,947 test)

## 🚀 Installation

### Prerequisites

- Python 3.12+ (recommended for TensorFlow compatibility)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DTSAR0/Human-Movement-Identification.git
cd Human-Movement-Identification
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter TensorFlow installation issues, see `SETUP_TENSORFLOW.md` for detailed instructions.

## 📁 Project Structure

```
Human-Movement-Identification/
├── Human_movement_identification.py  # Random Forest classifier
├── mlp_har_classifier.py            # Standalone MLP classifier
├── compare_rf_mlp.py                # Comparison script (RF vs MLP)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── SETUP_TENSORFLOW.md              # TensorFlow setup guide
└── human+activity+recognition+using+smartphones/
    └── UCI HAR Dataset/             # Dataset directory
```

## 🎯 Usage

### 1. Random Forest Classifier

Train and validate a Random Forest classifier:

```bash
python Human_movement_identification.py
```

Or with explicit argument:

```bash
python Human_movement_identification.py random_forest
```

Or in Python code:

```python
from Human_movement_identification import main_random_forest

model, metrics = main_random_forest()
```

**Note**: `Human_movement_identification.py` contains **only** Random Forest implementation. MLP and comparison functionality are in separate files.

**Random Forest Configuration:**
- Number of trees: **100** (decision trees)
- Max depth: None (unlimited)
- Random state: 42 (for reproducibility)

### 2. MLP (Multi-Layer Perceptron) Classifier

Train and validate an MLP neural network:

```bash
python mlp_har_classifier.py
```

Or in Python code:

```python
from mlp_har_classifier import main

main()
```

**MLP Architecture:**
- Input: 561 features
- Hidden Layer 1: 256 neurons + BatchNorm + Dropout(0.3)
- Hidden Layer 2: 128 neurons + BatchNorm + Dropout(0.3)
- Hidden Layer 3: 64 neurons + Dropout(0.2)
- Output: 6 classes (Softmax)
- Optimizer: Adam (learning_rate=1e-3)
- Loss: Categorical Cross-Entropy
- Regularization: Early Stopping, ReduceLROnPlateau

### 3. Model Comparison

Compare Random Forest and MLP models side-by-side:

```bash
python compare_rf_mlp.py
```

This script:
- Uses the **same train/test split** (70/30) for both models
- Applies StandardScaler preprocessing for MLP only
- Trains both models and compares their performance
- Generates comprehensive visualizations

**Comparison Features:**
- Overall metrics comparison (accuracy, F1, precision, recall)
- Per-class accuracy comparison
- Confusion matrix heatmaps for both models
- Training history visualization
- Train-test gap analysis

## 📊 Validation Metrics

Both classifiers automatically compute and display:

### Overall Metrics
- **Accuracy** - Overall classification accuracy (train and test)
- **F1 Score** - Macro and weighted averages
- **Precision** - Macro average precision
- **Recall** - Macro average recall
- **Train-Test Gap** - Difference between training and test accuracy (overfitting indicator)

### Per-Class Metrics
- **Per-Class Accuracy** - Accuracy for each activity class
- **Per-Class F1 Score** - F1 score for each activity class
- **Confusion Matrix** - Visual representation of classification errors
- **Classification Report** - Detailed per-class metrics (precision, recall, F1, support)

### Visualizations
- Confusion matrix heatmaps (using seaborn if available)
- Feature importance plots (Random Forest)
- Training history plots (MLP: accuracy and loss over epochs)
- Comparison bar charts (RF vs MLP)

## 🔬 Model Details

### Random Forest
- **Algorithm**: Ensemble of decision trees
- **Preprocessing**: None (works on raw features)
- **Advantages**: 
  - Fast training
  - Feature importance analysis
  - No hyperparameter tuning needed
  - Good baseline performance

### MLP (Feedforward Neural Network)
- **Algorithm**: Deep feedforward neural network
- **Preprocessing**: StandardScaler (normalization)
- **Advantages**:
  - Can learn complex non-linear patterns
  - Better generalization with proper regularization
  - Typically achieves higher accuracy on this dataset

## 📈 Expected Results

Based on typical runs:
- **Random Forest**: ~97-98% test accuracy
- **MLP**: ~98-99% test accuracy
- **Best Model**: Usually MLP (slightly better performance)

## 🛠️ Dependencies

Key libraries used:
- `numpy` - Numerical computations
- `scikit-learn` - Random Forest, metrics, preprocessing
- `tensorflow` - MLP neural network (optional)
- `matplotlib` - Plotting and visualization
- `seaborn` - Enhanced visualizations (optional)
- `joblib` - Model serialization

See `requirements.txt` for complete list.

## 🔧 Configuration

### Random Forest Parameters
Can be adjusted in `train_random_forest()` function or `main_random_forest()`:
```python
model = train_random_forest(
    X_train, y_train,
    n_estimators=100,    # Number of trees
    max_depth=None,      # Tree depth limit
    random_state=42,     # Reproducibility
    n_jobs=-1           # Parallel processing
)
```

### MLP Parameters
Can be adjusted in `mlp_har_classifier.py`:
```python
LEARNING_RATE = 1e-3
EPOCHS = 100
HIDDEN_NEURONS = 256
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
```

## 📝 Notes

- **Reproducibility**: All scripts use `random_state=42` for consistent results
- **Data Split**: Comparison script uses identical 70/30 train/test split for fair comparison
- **Preprocessing**: Only MLP uses feature scaling (StandardScaler); Random Forest works on raw features
- **Model Saving**: Trained models are saved as `.pkl` (Random Forest) and `.h5` (MLP) files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes.

## 👤 Author

DTSAR0

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the HAR dataset
- Scikit-learn and TensorFlow communities for excellent documentation

---

For detailed TensorFlow setup instructions, see `SETUP_TENSORFLOW.md`.
