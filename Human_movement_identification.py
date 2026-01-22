import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check for required libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, 
        confusion_matrix, 
        accuracy_score, 
        f1_score,
        precision_score,
        recall_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn is not installed. Install it: pip install scikit-learn")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn is not installed. Install it: pip install seaborn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow is not installed. For neural network training, install: pip install tensorflow")


def load_har_dataset(dataset_path):
    """
    Loads the UCI HAR dataset for classifier training
    
    Args:
        dataset_path: path to the 'UCI HAR Dataset' folder
    
    Returns:
        X_train, y_train, X_test, y_test, activity_names
    """
    dataset_path = Path(dataset_path)
    
    # Load training data
    X_train_path = dataset_path / "train" / "X_train.txt"
    y_train_path = dataset_path / "train" / "y_train.txt"
    
    # Load test data
    X_test_path = dataset_path / "test" / "X_test.txt"
    y_test_path = dataset_path / "test" / "y_test.txt"
    
    # Load activity labels
    activity_labels_path = dataset_path / "activity_labels.txt"
    
    print("Loading UCI HAR dataset...")
    X_train = np.loadtxt(X_train_path)
    y_train = np.loadtxt(y_train_path, dtype=int)
    X_test = np.loadtxt(X_test_path)
    y_test = np.loadtxt(y_test_path, dtype=int)
    
    activity_names = {}
    if activity_labels_path.exists():
        with open(activity_labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    activity_names[int(parts[0])] = ' '.join(parts[1:])
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}, unique classes: {np.unique(y_train)}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}, unique classes: {np.unique(y_test)}")
    
    # Convert labels from 1-6 to 0-5 for scikit-learn compatibility
    y_train = y_train - 1
    y_test = y_test - 1
    
    return X_train, y_train, X_test, y_test, activity_names


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42, n_jobs=-1):
    """
    Trains a Random Forest classifier
    
    Args:
        X_train: training data
        y_train: training labels
        n_estimators: number of trees in the forest (default 100)
        max_depth: maximum depth of the tree (None = no limit)
        random_state: seed for reproducibility
        n_jobs: number of cores for parallelization (-1 = all available)
    
    Returns:
        model: trained Random Forest classifier
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Install it: pip install scikit-learn")
    
    print(f"\n{'='*60}")
    print("Training Random Forest classifier...")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  - Number of trees: {n_estimators}")
    print(f"  - Maximum depth: {max_depth if max_depth else 'No limit'}")
    print(f"  - Random state: {random_state}")
    print(f"{'='*60}\n")
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("\nTraining completed!")
    
    return model


def validate_classifier(model, X_test, y_test, activity_names, show_plots=True):
    """
    Validates the classifier and displays metrics: confusion matrix, F1 score, accuracy
    
    Args:
        model: trained classifier
        X_test: test data
        y_test: test labels
        activity_names: dictionary with activity names
        show_plots: whether to show plots (default True)
    
    Returns:
        dict: dictionary with metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Install it: pip install scikit-learn")
    
    print(f"\n{'='*60}")
    print("Classifier Validation")
    print(f"{'='*60}\n")
    
    # Predictions
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\n📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📈 F1 Score (Macro Average): {f1_macro:.4f}")
    print(f"📈 F1 Score (Weighted Average): {f1_weighted:.4f}")
    print(f"\n📉 Precision (Macro Average): {precision_macro:.4f}")
    print(f"📉 Recall (Macro Average): {recall_macro:.4f}")
    
    # F1 Score per class
    print(f"\n{'='*60}")
    print("F1 Score per class:")
    print(f"{'='*60}")
    class_names = [activity_names.get(i+1, f"Class {i}") for i in range(len(activity_names))]
    for i, (class_name, f1) in enumerate(zip(class_names, f1_per_class)):
        print(f"  {class_name:20s}: {f1:.4f}")
    
    # Classification Report
    print(f"\n{'='*60}")
    print("Classification Report:")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix visualization
    if show_plots:
        plt.figure(figsize=(12, 10))
        if SEABORN_AVAILABLE:
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Number of samples'}
            )
        else:
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.colorbar(label='Number of samples')
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
            plt.yticks(range(len(class_names)), class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10)
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Labels', fontsize=12)
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm
    }
    
    return metrics


def plot_feature_importance(model, top_n=20, feature_names=None):
    """
    Visualizes feature importance for Random Forest
    
    Args:
        model: trained Random Forest classifier
        top_n: number of top features to display
        feature_names: list of feature names (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices][::-1], color='steelblue', edgecolor='black')
    
    if feature_names is not None:
        labels = [f"Feature {indices[i]}" if feature_names[indices[i]] == '' 
                 else feature_names[indices[i]] for i in range(top_n-1, -1, -1)]
    else:
        labels = [f"Feature {indices[i]}" for i in range(top_n-1, -1, -1)]
    
    plt.yticks(range(top_n), labels)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def main_random_forest():
    """
    Main function for training and validating Random Forest classifier
    Combines train and test data, then splits into 70% training / 30% testing
    """
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is not installed!")
        print("Install dependencies: pip install -r requirements.txt")
        return
    
    # Dataset path
    dataset_path = Path("human+activity+recognition+using+smartphones/UCI HAR Dataset")
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at path: {dataset_path}")
        print("Please check the dataset path.")
        return
    
    # Load data
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, activity_names = load_har_dataset(dataset_path)
    
    # Combine train and test data
    print(f"\n{'='*60}")
    print("Combining data...")
    print(f"{'='*60}")
    X_all = np.vstack([X_train_orig, X_test_orig])
    y_all = np.hstack([y_train_orig, y_test_orig])
    print(f"Total number of samples: {X_all.shape[0]}")
    print(f"Number of features: {X_all.shape[1]}")
    
    # Split into 70% training / 30% testing
    print(f"\n{'='*60}")
    print("Splitting dataset: 70% training / 30% testing")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, 
        y_all, 
        test_size=0.3, 
        random_state=42, 
        stratify=y_all  # Preserves class proportions
    )
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X_all.shape[0]*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X_all.shape[0]*100:.1f}%)")
    
    # Train Random Forest
    # Parameters can be adjusted:
    # n_estimators - more trees = better quality, but slower training
    # max_depth - depth limit helps avoid overfitting
    model = train_random_forest(
        X_train, 
        y_train, 
        n_estimators=100,  # Number of trees
        max_depth=None,    # No depth limit
        random_state=42,
        n_jobs=-1          # Use all available cores
    )
    
    # Validate classifier
    metrics = validate_classifier(model, X_test, y_test, activity_names, show_plots=True)
    
    # Feature importance visualization
    print(f"\n{'='*60}")
    print("Feature importance visualization...")
    print(f"{'='*60}\n")
    plot_feature_importance(model, top_n=20)
    
    # Save model (optional)
    try:
        import joblib
        model_path = "random_forest_model.pkl"
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved as '{model_path}'")
    except ImportError:
        print("\n💡 To save the model, install joblib: pip install joblib")
    
    print(f"\n{'='*60}")
    print("Training and validation completed!")
    print(f"{'='*60}\n")
    
    return model, metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "random_forest":
        # Run Random Forest classifier
        main_random_forest()
    else:
        print("Usage:")
        print("  python Human_movement_identification.py random_forest")
        print("\nOr call the main_random_forest() function in code.")
