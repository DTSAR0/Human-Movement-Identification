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


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed: random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


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


def validate_classifier(model, X_test, y_test, activity_names, X_train=None, y_train=None, show_plots=True):
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
    
    # Training accuracy if provided
    accuracy_train = None
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division='warn')
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division='warn')
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division='warn')  # type: ignore
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division='warn')
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division='warn')
    
    # Accuracy per class
    accuracy_per_class = []
    num_classes = int(np.max(y_test)) + 1
    class_names = [activity_names.get(i + 1, f"Class {i}") for i in range(num_classes)]  # type: ignore

    for i in range(len(class_names)):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            accuracy_per_class.append(class_acc)
        else:
            accuracy_per_class.append(0.0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if accuracy_train is not None:
        print(f"📊 Training Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
        accuracy_diff = abs(accuracy_train - accuracy)
        print(f"📊 Accuracy Difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
    print(f"\n📈 F1 Score (Macro Average): {f1_macro:.4f}")
    print(f"📈 F1 Score (Weighted Average): {f1_weighted:.4f}")
    print(f"\n📉 Precision (Macro Average): {precision_macro:.4f}")
    print(f"📉 Recall (Macro Average): {recall_macro:.4f}")
    
    # Per-class metrics
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    print(f"{'Class':<25} {'Accuracy':<15} {'F1 Score':<15}")
    print("-" * 55)
    # Ensure f1_per_class is a list/array for len() check
    # f1_score with average=None always returns an array, but type checker doesn't know this
    if isinstance(f1_per_class, np.ndarray):
        f1_per_class_list = f1_per_class.tolist()
    elif isinstance(f1_per_class, (list, tuple)):
        f1_per_class_list = list(f1_per_class)
    else:
        f1_per_class_list = []  # type: ignore
    
    for i, class_name in enumerate(class_names):
        if i < len(accuracy_per_class) and i < len(f1_per_class_list):
            print(f"{class_name:<25} {accuracy_per_class[i]:<15.4f} {f1_per_class_list[i]:<15.4f}")
    
    # F1 Score per class
    print(f"\n{'='*60}")
    print("F1 Score per class:")
    print(f"{'='*60}")
    for i, (class_name, f1) in enumerate(zip(class_names, f1_per_class)):  # type: ignore
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
    # Convert f1_per_class to list for storage
    if isinstance(f1_per_class, np.ndarray):
        f1_per_class_stored = f1_per_class.tolist()
    elif isinstance(f1_per_class, (list, tuple)):
        f1_per_class_stored = list(f1_per_class)
    else:
        f1_per_class_stored = []  # type: ignore
    
    metrics = {
        'accuracy_test': accuracy,
        'accuracy': accuracy,  # For backward compatibility
        'accuracy_train': accuracy_train,
        'accuracy_per_class': accuracy_per_class,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class_stored,
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
    
    # Use original train/test split (NO COMBINING - preserves subject separation)
    print(f"\n{'='*60}")
    print("USING ORIGINAL DATASET SPLIT (NO COMBINING)")
    print(f"{'='*60}")
    print("Train: Original UCI HAR train set")
    print("Test:  Original UCI HAR test set")
    print(f"Training set: {X_train_orig.shape[0]} samples")  # type: ignore
    print(f"Test set: {X_test_orig.shape[0]} samples")  # type: ignore
    print("Note: Preserves original subject separation")
    
    # Train Random Forest on original train set
    # Parameters can be adjusted:
    # n_estimators - more trees = better quality, but slower training
    # max_depth - depth limit helps avoid overfitting
    model = train_random_forest(
        X_train_orig, 
        y_train_orig, 
        n_estimators=100,  # Number of trees
        max_depth=None,    # No depth limit
        random_state=42,
        n_jobs=-1          # Use all available cores
    )
    
    # Validate classifier on original test set (with training data for training accuracy)
    metrics = validate_classifier(model, X_test_orig, y_test_orig, activity_names, X_train=X_train_orig, y_train=y_train_orig, show_plots=True)
    
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


def compare_models(rf_metrics, mlp_metrics, activity_names):
    """
    Порівнює результати Random Forest та MLP моделей
    
    Args:
        rf_metrics: словник з метриками Random Forest
        mlp_metrics: словник з метриками MLP
        activity_names: словник з назвами активностей
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON: Random Forest vs MLP")
    print(f"{'='*80}\n")
    
    class_names = [activity_names.get(i+1, f"Class {i}") for i in range(len(activity_names))]
    
    # Загальні метрики
    print(f"{'Metric':<30} {'Random Forest':<20} {'MLP':<20} {'Difference':<15}")
    print("-" * 85)
    
    # Test Accuracy
    rf_acc = rf_metrics.get('accuracy_test', rf_metrics.get('accuracy', 0))
    mlp_acc = mlp_metrics.get('test_accuracy', 0)
    diff_acc = mlp_acc - rf_acc
    print(f"{'Test Accuracy':<30} {rf_acc:<20.4f} {mlp_acc:<20.4f} {diff_acc:+.4f} ({diff_acc*100:+.2f}%)")
    
    # Training Accuracy
    rf_train_acc = rf_metrics.get('accuracy_train', None)
    mlp_train_acc = mlp_metrics.get('train_accuracy', None)
    if rf_train_acc is not None and mlp_train_acc is not None:
        diff_train = mlp_train_acc - rf_train_acc
        print(f"{'Training Accuracy':<30} {rf_train_acc:<20.4f} {mlp_train_acc:<20.4f} {diff_train:+.4f} ({diff_train*100:+.2f}%)")
    
    # F1 Macro
    rf_f1 = rf_metrics.get('f1_macro', 0)
    mlp_f1 = mlp_metrics.get('f1_macro', 0)
    diff_f1 = mlp_f1 - rf_f1
    print(f"{'F1 Score (Macro)':<30} {rf_f1:<20.4f} {mlp_f1:<20.4f} {diff_f1:+.4f}")
    
    # F1 Weighted
    rf_f1w = rf_metrics.get('f1_weighted', 0)
    mlp_f1w = mlp_metrics.get('f1_weighted', 0)
    diff_f1w = mlp_f1w - rf_f1w
    print(f"{'F1 Score (Weighted)':<30} {rf_f1w:<20.4f} {mlp_f1w:<20.4f} {diff_f1w:+.4f}")
    
    # Precision
    rf_prec = rf_metrics.get('precision_macro', 0)
    mlp_prec = mlp_metrics.get('precision_macro', 0)
    diff_prec = mlp_prec - rf_prec
    print(f"{'Precision (Macro)':<30} {rf_prec:<20.4f} {mlp_prec:<20.4f} {diff_prec:+.4f}")
    
    # Recall
    rf_rec = rf_metrics.get('recall_macro', 0)
    mlp_rec = mlp_metrics.get('recall_macro', 0)
    diff_rec = mlp_rec - rf_rec
    print(f"{'Recall (Macro)':<30} {rf_rec:<20.4f} {mlp_rec:<20.4f} {diff_rec:+.4f}")
    
    print("\n" + "=" * 85)
    
    # Per-class accuracy comparison
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Class':<25} {'Random Forest':<20} {'MLP':<20} {'Difference':<15}")
    print("-" * 80)
    
    rf_acc_per_class = rf_metrics.get('accuracy_per_class', [])
    mlp_acc_per_class = mlp_metrics.get('accuracy_per_class', [])
    
    if len(mlp_acc_per_class) == 0:
        # Якщо MLP не має accuracy_per_class, обчислимо
        mlp_acc_per_class = []
        # Потрібно буде передати confusion matrix або predictions
        pass
    
    for i, class_name in enumerate(class_names):
        if i < len(rf_acc_per_class) and i < len(mlp_acc_per_class):
            rf_class_acc = rf_acc_per_class[i]
            mlp_class_acc = mlp_acc_per_class[i]
            diff = mlp_class_acc - rf_class_acc
            print(f"{class_name:<25} {rf_class_acc:<20.4f} {mlp_class_acc:<20.4f} {diff:+.4f} ({diff*100:+.2f}%)")
    
    print("\n" + "=" * 80)
    
    # Per-class F1 comparison
    print(f"\n{'='*80}")
    print("PER-CLASS F1 SCORE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Class':<25} {'Random Forest':<20} {'MLP':<20} {'Difference':<15}")
    print("-" * 80)
    
    rf_f1_per_class = rf_metrics.get('f1_per_class', [])
    mlp_f1_per_class = mlp_metrics.get('f1_per_class', [])
    
    # Конвертація в список якщо потрібно
    if isinstance(rf_f1_per_class, np.ndarray):
        rf_f1_per_class = rf_f1_per_class.tolist()
    elif not isinstance(rf_f1_per_class, list):
        rf_f1_per_class = list(rf_f1_per_class) if hasattr(rf_f1_per_class, '__iter__') else []
    
    if isinstance(mlp_f1_per_class, np.ndarray):
        mlp_f1_per_class = mlp_f1_per_class.tolist()
    elif not isinstance(mlp_f1_per_class, list):
        mlp_f1_per_class = list(mlp_f1_per_class) if hasattr(mlp_f1_per_class, '__iter__') else []
    
    for i, class_name in enumerate(class_names):
        if i < len(rf_f1_per_class) and i < len(mlp_f1_per_class):
            rf_class_f1 = rf_f1_per_class[i]
            mlp_class_f1 = mlp_f1_per_class[i]
            diff = mlp_class_f1 - rf_class_f1
            print(f"{class_name:<25} {rf_class_f1:<20.4f} {mlp_class_f1:<20.4f} {diff:+.4f}")
    
    print("\n" + "=" * 80)
    
    # Підсумок
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    winners = {
        'Test Accuracy': 'MLP' if mlp_acc > rf_acc else 'Random Forest',
        'F1 Score (Macro)': 'MLP' if mlp_f1 > rf_f1 else 'Random Forest',
        'F1 Score (Weighted)': 'MLP' if mlp_f1w > rf_f1w else 'Random Forest',
        'Precision': 'MLP' if mlp_prec > rf_prec else 'Random Forest',
        'Recall': 'MLP' if mlp_rec > rf_rec else 'Random Forest',
    }
    
    rf_wins = sum(1 for v in winners.values() if v == 'Random Forest')
    mlp_wins = sum(1 for v in winners.values() if v == 'MLP')
    
    print(f"\nRandom Forest wins: {rf_wins} metrics")
    print(f"MLP wins: {mlp_wins} metrics")
    
    if mlp_wins > rf_wins:
        print("\n🏆 Overall Winner: MLP")
    elif rf_wins > mlp_wins:
        print("\n🏆 Overall Winner: Random Forest")
    else:
        print("\n🤝 Models are comparable")
    
    print(f"\n{'='*80}\n")
    
    # Візуалізація порівняння
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Загальні метрики
    metrics_names = ['Test Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
    rf_values = [rf_acc, rf_f1, rf_f1w, rf_prec, rf_rec]
    mlp_values = [mlp_acc, mlp_f1, mlp_f1w, mlp_prec, mlp_rec]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, rf_values, width, label='Random Forest', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, mlp_values, width, label='MLP', color='coral', alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Overall Metrics Comparison', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.1])
    
    # Per-class accuracy
    if len(rf_acc_per_class) > 0 and len(mlp_acc_per_class) > 0:
        x_class = np.arange(len(class_names))
        axes[0, 1].bar(x_class - width/2, rf_acc_per_class, width, label='Random Forest', color='steelblue', alpha=0.8)
        axes[0, 1].bar(x_class + width/2, mlp_acc_per_class, width, label='MLP', color='coral', alpha=0.8)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Per-Class Accuracy Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x_class)
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1.1])
    
    # Per-class F1
    if len(rf_f1_per_class) > 0 and len(mlp_f1_per_class) > 0:
        axes[1, 0].bar(x_class - width/2, rf_f1_per_class, width, label='Random Forest', color='steelblue', alpha=0.8)
        axes[1, 0].bar(x_class + width/2, mlp_f1_per_class, width, label='MLP', color='coral', alpha=0.8)
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Per-Class F1 Score Comparison', fontweight='bold')
        axes[1, 0].set_xticks(x_class)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.1])
    
    # Confusion matrices
    rf_cm = rf_metrics.get('confusion_matrix', None)
    mlp_cm = mlp_metrics.get('confusion_matrix', None)
    
    if rf_cm is not None:
        if SEABORN_AVAILABLE:
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], 
                       xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
        else:
            axes[1, 1].imshow(rf_cm, cmap='Blues', interpolation='nearest')
            axes[1, 1].set_xticks(range(len(class_names)))
            axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1, 1].set_yticks(range(len(class_names)))
            axes[1, 1].set_yticklabels(class_names)
        
        axes[1, 1].set_title('Random Forest - Confusion Matrix', fontweight='bold')
        axes[1, 1].set_ylabel('True Labels')
        axes[1, 1].set_xlabel('Predicted Labels')
    
    plt.tight_layout()
    plt.show()


def main_compare_models():
    """
    Головна функція для навчання та порівняння Random Forest та MLP моделей
    """
    import sys
    
    print("="*80)
    print("COMPARING RANDOM FOREST AND MLP MODELS")
    print("="*80)
    
    # Перевірка залежностей
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is not installed!")
        return
    
    # Перевірка TensorFlow для MLP
    try:
        import tensorflow as tf  # type: ignore
        from mlp_har_classifier import (
            load_har_dataset as load_har_mlp,
            build_mlp_model,
            train_mlp,
            evaluate_model as evaluate_mlp
        )
        from tensorflow.keras.utils import to_categorical  # type: ignore
        from sklearn.preprocessing import StandardScaler
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        print("Error: TensorFlow is not installed or mlp_har_classifier.py not found!")
        print("Install TensorFlow: pip install tensorflow")
        return
    
    # Шлях до датасету
    dataset_path = Path("human+activity+recognition+using+smartphones/UCI HAR Dataset")
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at path: {dataset_path}")
        return
    
    # Завантаження даних
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, activity_names = load_har_dataset(dataset_path)
    
    # Використовуємо оригінальний train/test split (БЕЗ об'єднання - зберігаємо розділення людей)
    print(f"\n{'='*60}")
    print("USING ORIGINAL DATASET SPLIT (NO COMBINING)")
    print(f"{'='*60}")
    print("Train: Original UCI HAR train set")
    print("Test:  Original UCI HAR test set")
    print(f"Train: {X_train_orig.shape[0]} samples")  # type: ignore
    print(f"Test:  {X_test_orig.shape[0]} samples")  # type: ignore
    print("Note: Both RF and MLP use the SAME original train/test split")
    print("      Preserves original subject separation")
    
    # ========== RANDOM FOREST ==========
    print(f"\n{'='*80}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*80}\n")
    
    # Навчання Random Forest на оригінальному train set
    rf_model = train_random_forest(
        X_train_orig, y_train_orig,
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    # Валідація Random Forest на оригінальному test set
    rf_metrics = validate_classifier(
        rf_model, X_test_orig, y_test_orig, activity_names,
        X_train=X_train_orig, y_train=y_train_orig,
        show_plots=False  # Не показувати графіки під час порівняння
    )
    
    # ========== MLP ==========
    print(f"\n{'='*80}")
    print("TRAINING MLP")
    print(f"{'='*80}\n")
    
    # Встановлення seed
    tf.keras.utils.set_random_seed(42)  # type: ignore
    
    # Нормалізація для MLP (fit тільки на train!)
    print("Normalizing features for MLP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    
    # Validation split для MLP: 80% train / 20% val (з оригінального train set)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_orig,
        test_size=0.2,
        random_state=42,
        stratify=y_train_orig
    )
    print(f"MLP Train: {X_train_final.shape[0]} samples (80% of original train)")  # type: ignore
    print(f"MLP Val:   {X_val.shape[0]} samples (20% of original train)")  # type: ignore
    print(f"MLP Test:  {X_test_scaled.shape[0]} samples (original test, same as RF)")  # type: ignore
    
    # Конвертація міток в one-hot для MLP
    num_classes = int(np.max(np.concatenate([y_train_orig, y_test_orig]))) + 1
    y_train_final_cat = to_categorical(y_train_final, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_orig, num_classes=num_classes)
    
    # Побудова та навчання MLP
    mlp_model = build_mlp_model(
        input_shape=X_train_final.shape[1],  # type: ignore
        num_classes=num_classes,
        hidden_neurons=256,
        learning_rate=1e-3,
        dropout_rate=0.3
    )
    
    mlp_history = train_mlp(
        mlp_model,
        X_train_final, y_train_final_cat,
        X_val, y_val_cat,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Оцінка MLP
    mlp_metrics = evaluate_mlp(
        mlp_model,
        X_train_final, y_train_final_cat,
        X_test_scaled, y_test_cat,
        activity_names
    )
    
    # Додавання accuracy_per_class та f1_per_class до mlp_metrics якщо їх немає
    num_classes_for_names = int(np.max(np.concatenate([y_train_orig, y_test_orig]))) + 1
    class_names_list = [activity_names.get(i+1, f"Class {i}") for i in range(num_classes_for_names)]
    
    mlp_cm = mlp_metrics.get('confusion_matrix', None)
    if mlp_cm is not None:
        # Обчислення accuracy_per_class
        if 'accuracy_per_class' not in mlp_metrics:
            mlp_acc_per_class = []
            for i in range(len(class_names_list)):
                if np.sum(mlp_cm[:, i]) > 0:
                    acc = mlp_cm[i, i] / np.sum(mlp_cm[:, i])
                    mlp_acc_per_class.append(acc)
                else:
                    mlp_acc_per_class.append(0.0)
            mlp_metrics['accuracy_per_class'] = mlp_acc_per_class
        
        # Обчислення f1_per_class
        if 'f1_per_class' not in mlp_metrics:
            from sklearn.metrics import f1_score
            y_test_labels = np.argmax(y_test_cat, axis=1)
            y_test_pred_proba = mlp_model.predict(X_test_scaled, verbose=0)
            y_test_pred = np.argmax(y_test_pred_proba, axis=1)
            mlp_f1_per_class = f1_score(y_test_labels, y_test_pred, average=None, zero_division='warn')  # type: ignore
            mlp_metrics['f1_per_class'] = mlp_f1_per_class
    
    # ========== ПОРІВНЯННЯ ==========
    compare_models(rf_metrics, mlp_metrics, activity_names)
    
    # Збереження моделей
    try:
        import joblib
        joblib.dump(rf_model, "random_forest_model.pkl")
        mlp_model.save("mlp_model.h5")
        print("✅ Both models saved successfully")
    except Exception as e:
        print(f"⚠️  Error saving models: {e}")
    
    return rf_model, mlp_model, rf_metrics, mlp_metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "random_forest":
        # Run Random Forest classifier only
        print("="*80)
        print("RUNNING RANDOM FOREST CLASSIFIER")
        print("="*80)
        main_random_forest()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare Random Forest and MLP
        main_compare_models()
    else:
        print("Usage:")
        print("  python Human_movement_identification.py random_forest  # Random Forest only")
        print("  python Human_movement_identification.py compare        # Compare RF and MLP")
        print("\nOr call the respective function in code.")
