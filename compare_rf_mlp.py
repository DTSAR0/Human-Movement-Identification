"""
Comparison of Random Forest and MLP models on UCI HAR Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# TensorFlow/Keras imports (only when needed)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow is not installed. MLP training will be skipped.")

# Seaborn for better plots (optional)
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TENSORFLOW_AVAILABLE:
        try:
            tf.keras.utils.set_random_seed(seed)  # type: ignore
        except Exception:
            pass


# ============================================================================
# DATA LOADING
# ============================================================================

def load_har_dataset(dataset_path):
    """
    Load UCI HAR dataset
    
    Args:
        dataset_path: path to 'UCI HAR Dataset' folder (str or Path)
    
    Returns:
        X_train, y_train, X_test, y_test, activity_names
    """
    dataset_path = Path(dataset_path)
    
    # File paths
    X_train_path = dataset_path / "train" / "X_train.txt"
    y_train_path = dataset_path / "train" / "y_train.txt"
    X_test_path = dataset_path / "test" / "X_test.txt"
    y_test_path = dataset_path / "test" / "y_test.txt"
    activity_labels_path = dataset_path / "activity_labels.txt"
    
    # Load data
    print("Loading UCI HAR dataset...")
    X_train = np.loadtxt(X_train_path)
    y_train = np.loadtxt(y_train_path, dtype=int)
    X_test = np.loadtxt(X_test_path)
    y_test = np.loadtxt(y_test_path, dtype=int)
    
    # Convert labels from 1-6 to 0-5
    y_train = y_train - 1
    y_test = y_test - 1
    
    # Load activity names
    activity_names = {}
    with open(activity_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label_id = int(parts[0])
                label_name = ' '.join(parts[1:])
                activity_names[label_id] = label_name
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, unique classes: {np.unique(y_train)}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, unique classes: {np.unique(y_test)}")
    
    return X_train, y_train, X_test, y_test, activity_names


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_rf(X_train, y_train, n_estimators=100, max_depth=None, random_state=42, n_jobs=-1):
    """
    Train Random Forest classifier
    
    Args:
        X_train: training features
        y_train: training labels
        n_estimators: number of trees
        max_depth: maximum depth of trees
        random_state: random seed
        n_jobs: number of parallel jobs
    
    Returns:
        trained model
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}")
    print("Training...")
    
    model.fit(X_train, y_train)
    
    print("Training completed!")
    
    return model


def train_mlp(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              learning_rate=1e-3, dropout_rate=0.3, verbose=1):
    """
    Train MLP model
    
    Args:
        X_train: training features (scaled)
        y_train: training labels (one-hot encoded)
        X_val: validation features (scaled)
        y_val: validation labels (one-hot encoded)
        epochs: number of epochs
        batch_size: batch size
        learning_rate: learning rate for Adam
        dropout_rate: dropout rate
        verbose: verbosity level
    
    Returns:
        trained model, training history
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed!")
    
    print("\n" + "="*60)
    print("TRAINING MLP")
    print("="*60)
    
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        
        # Layer 1
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Layer 2
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Layer 3
        layers.Dense(64, activation='relu', name='dense_3'),
        layers.Dropout(0.2, name='dropout_3'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=verbose
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=verbose
    )
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose
    )
    
    print("Training completed!")
    
    return model, history


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_sklearn_model(model, X_train, y_train, X_test, y_test, activity_names):
    """
    Evaluate scikit-learn model (Random Forest)
    
    Args:
        model: trained model
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
        activity_names: dictionary mapping label IDs to names
    
    Returns:
        dictionary with metrics
    """
    print("\n" + "="*60)
    print("EVALUATING RANDOM FOREST")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_test_gap = train_accuracy - test_accuracy
    
    f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division='warn')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division='warn')
    precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division='warn')
    recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division='warn')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Per-class accuracy
    num_classes = int(np.max(y_test)) + 1
    accuracy_per_class = []
    for i in range(num_classes):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_test_pred[mask])
            accuracy_per_class.append(acc)
        else:
            accuracy_per_class.append(0.0)
    
    # Class names
    class_names = [activity_names.get(i+1, f"Class {i}") for i in range(num_classes)]
    
    # Print results
    print(f"\nTrain Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Train-Test Gap: {train_test_gap:.4f} ({train_test_gap*100:.2f}%)")
    print(f"\nF1 Score (Macro):     {f1_macro:.4f}")
    print(f"F1 Score (Weighted):  {f1_weighted:.4f}")
    print(f"Precision (Macro):    {precision_macro:.4f}")
    print(f"Recall (Macro):       {recall_macro:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division='warn'))
    
    # Return metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_test_gap': train_test_gap,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'accuracy_per_class': accuracy_per_class,
        'class_names': class_names,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }
    
    return metrics


def evaluate_keras_model(model, X_train, y_train, X_test, y_test, activity_names):
    """
    Evaluate Keras model (MLP)
    
    Args:
        model: trained model
        X_train: training features (scaled)
        y_train: training labels (one-hot encoded)
        X_test: test features (scaled)
        y_test: test labels (one-hot encoded)
        activity_names: dictionary mapping label IDs to names
    
    Returns:
        dictionary with metrics
    """
    print("\n" + "="*60)
    print("EVALUATING MLP")
    print("="*60)
    
    # Predictions
    y_train_pred_proba = model.predict(X_train, verbose=0)
    y_test_pred_proba = model.predict(X_test, verbose=0)
    
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Metrics
    train_accuracy = accuracy_score(y_train_labels, y_train_pred)
    test_accuracy = accuracy_score(y_test_labels, y_test_pred)
    train_test_gap = train_accuracy - test_accuracy
    
    f1_macro = f1_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    f1_weighted = f1_score(y_test_labels, y_test_pred, average='weighted', zero_division='warn')
    precision_macro = precision_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    recall_macro = recall_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_test_pred)
    
    # Per-class accuracy
    num_classes = int(np.max(y_test_labels)) + 1
    accuracy_per_class = []
    for i in range(num_classes):
        mask = y_test_labels == i
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test_labels[mask], y_test_pred[mask])
            accuracy_per_class.append(acc)
        else:
            accuracy_per_class.append(0.0)
    
    # Class names
    class_names = [activity_names.get(i+1, f"Class {i}") for i in range(num_classes)]
    
    # Print results
    print(f"\nTrain Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Train-Test Gap: {train_test_gap:.4f} ({train_test_gap*100:.2f}%)")
    print(f"\nF1 Score (Macro):     {f1_macro:.4f}")
    print(f"F1 Score (Weighted):  {f1_weighted:.4f}")
    print(f"Precision (Macro):    {precision_macro:.4f}")
    print(f"Recall (Macro):       {recall_macro:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_labels, y_test_pred, target_names=class_names, zero_division='warn'))
    
    # Return metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_test_gap': train_test_gap,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'accuracy_per_class': accuracy_per_class,
        'class_names': class_names,
        'y_test': y_test_labels,
        'y_test_pred': y_test_pred
    }
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, class_names, title, ax=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: confusion matrix
        class_names: list of class names
        title: plot title
        ax: matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if SEABORN_AVAILABLE:
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={'label': 'Count'}
        )
    else:
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Count')
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')


def compare_metrics(rf_metrics, mlp_metrics):
    """
    Compare metrics between Random Forest and MLP
    
    Args:
        rf_metrics: metrics dictionary for Random Forest
        mlp_metrics: metrics dictionary for MLP
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON: Random Forest vs MLP")
    print("="*80)
    
    # Overall metrics table
    print(f"\n{'Metric':<30} {'Random Forest':<20} {'MLP':<20} {'Diff':<15}")
    print("-" * 85)
    
    metrics_to_compare = [
        ('Test Accuracy', 'test_accuracy'),
        ('F1 Score (Macro)', 'f1_macro'),
        ('F1 Score (Weighted)', 'f1_weighted'),
        ('Precision (Macro)', 'precision_macro'),
        ('Recall (Macro)', 'recall_macro')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        rf_val = rf_metrics.get(metric_key, 0)
        mlp_val = mlp_metrics.get(metric_key, 0)
        diff = mlp_val - rf_val
        print(f"{metric_name:<30} {rf_val:<20.4f} {mlp_val:<20.4f} {diff:+.4f}")
    
    # Per-class accuracy comparison
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Class':<25} {'Random Forest':<20} {'MLP':<20} {'Diff':<15}")
    print("-" * 80)
    
    class_names = rf_metrics.get('class_names', [])
    rf_acc_per_class = rf_metrics.get('accuracy_per_class', [])
    mlp_acc_per_class = mlp_metrics.get('accuracy_per_class', [])
    
    for i, class_name in enumerate(class_names):
        if i < len(rf_acc_per_class) and i < len(mlp_acc_per_class):
            rf_acc = rf_acc_per_class[i]
            mlp_acc = mlp_acc_per_class[i]
            diff = mlp_acc - rf_acc
            print(f"{class_name:<25} {rf_acc:<20.4f} {mlp_acc:<20.4f} {diff:+.4f} ({diff*100:+.2f}%)")
    
    print("\n" + "="*80)
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Overall metrics bar chart
    ax1 = plt.subplot(2, 3, 1)
    metric_names = [m[0] for m in metrics_to_compare]
    rf_values = [rf_metrics.get(m[1], 0) for m in metrics_to_compare]
    mlp_values = [mlp_metrics.get(m[1], 0) for m in metrics_to_compare]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax1.bar(x - width/2, rf_values, width, label='Random Forest', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, mlp_values, width, label='MLP', color='coral', alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)
    
    # 2. Per-class accuracy comparison
    ax2 = plt.subplot(2, 3, 2)
    x_class = np.arange(len(class_names))
    ax2.bar(x_class - width/2, rf_acc_per_class, width, label='Random Forest', color='steelblue', alpha=0.8)
    ax2.bar(x_class + width/2, mlp_acc_per_class, width, label='MLP', color='coral', alpha=0.8)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy Comparison', fontweight='bold')
    ax2.set_xticks(x_class)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    # 3. Random Forest confusion matrix
    ax3 = plt.subplot(2, 3, 3)
    plot_confusion_matrix(
        rf_metrics['confusion_matrix'],
        class_names,
        'Random Forest - Confusion Matrix',
        ax=ax3
    )
    
    # 4. MLP confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    plot_confusion_matrix(
        mlp_metrics['confusion_matrix'],
        class_names,
        'MLP - Confusion Matrix',
        ax=ax4
    )
    
    # 5. Train-Test Gap comparison
    ax5 = plt.subplot(2, 3, 5)
    gap_data = [
        rf_metrics.get('train_test_gap', 0),
        mlp_metrics.get('train_test_gap', 0)
    ]
    ax5.bar(['Random Forest', 'MLP'], gap_data, color=['steelblue', 'coral'], alpha=0.8)
    ax5.set_ylabel('Train-Test Gap')
    ax5.set_title('Train-Test Accuracy Gap', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    SUMMARY
    
    Random Forest:
    - Test Accuracy: {rf_metrics['test_accuracy']:.4f}
    - F1 Macro: {rf_metrics['f1_macro']:.4f}
    - Train-Test Gap: {rf_metrics['train_test_gap']:.4f}
    
    MLP:
    - Test Accuracy: {mlp_metrics['test_accuracy']:.4f}
    - F1 Macro: {mlp_metrics['f1_macro']:.4f}
    - Train-Test Gap: {mlp_metrics['train_test_gap']:.4f}
    
    Winner:
    """
    
    if mlp_metrics['test_accuracy'] > rf_metrics['test_accuracy']:
        summary_text += "🏆 MLP (by test accuracy)"
    elif rf_metrics['test_accuracy'] > mlp_metrics['test_accuracy']:
        summary_text += "🏆 Random Forest (by test accuracy)"
    else:
        summary_text += "🤝 Tie"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             family='monospace', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run comparison"""
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Dataset path
    dataset_path = "human+activity+recognition+using+smartphones/UCI HAR Dataset"
    
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, activity_names = load_har_dataset(dataset_path)
    
    # Use original train/test split (NO COMBINING - preserves subject separation)
    print("\n" + "="*80)
    print("USING ORIGINAL DATASET SPLIT (NO COMBINING)")
    print("="*80)
    print("Train: Original UCI HAR train set")
    print("Test:  Original UCI HAR test set")
    print(f"Train: {X_train_orig.shape[0]} samples")  # type: ignore
    print(f"Test:  {X_test_orig.shape[0]} samples")  # type: ignore
    print("Note: Both RF and MLP use the SAME original train/test split")
    
    # ========== RANDOM FOREST ==========
    # Train on original train set, test on original test set
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    rf_model = train_rf(X_train_orig, y_train_orig)
    
    # Evaluate on original test set
    rf_metrics = evaluate_sklearn_model(rf_model, X_train_orig, y_train_orig, X_test_orig, y_test_orig, activity_names)
    
    # ========== MLP ==========
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠️  TensorFlow not available. Skipping MLP training.")
        return
    
    # Scale features for MLP (fit only on train!)
    print("\n" + "="*80)
    print("PREPROCESSING FOR MLP")
    print("="*80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    print("Features scaled using StandardScaler (fit on train, transform on test)")
    
    # Validation split for MLP: 80% train / 20% val (from original train set)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_orig,
        test_size=0.2,
        random_state=42,
        stratify=y_train_orig
    )
    print(f"MLP Train: {X_train_final.shape[0]} samples (80% of original train)")  # type: ignore
    print(f"MLP Val:   {X_val.shape[0]} samples (20% of original train)")  # type: ignore
    print(f"MLP Test:  {X_test_scaled.shape[0]} samples (original test, same as RF)")  # type: ignore
    
    # One-hot encode labels
    num_classes = int(np.max(np.concatenate([y_train_orig, y_test_orig]))) + 1
    y_train_final_cat = to_categorical(y_train_final, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_orig, num_classes=num_classes)
    
    # Train MLP
    mlp_model, mlp_history = train_mlp(
        X_train_final, y_train_final_cat,
        X_val, y_val_cat,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        dropout_rate=0.3,
        verbose=1
    )
    
    # Evaluate MLP (use full train set for train accuracy)
    # Re-scale full train set for evaluation
    X_train_full_scaled = scaler.transform(X_train_orig)
    y_train_full_cat = to_categorical(y_train_orig, num_classes=num_classes)
    
    mlp_metrics = evaluate_keras_model(
        mlp_model,
        X_train_full_scaled, y_train_full_cat,
        X_test_scaled, y_test_cat,
        activity_names
    )
    
    # ========== COMPARISON ==========
    compare_metrics(rf_metrics, mlp_metrics)
    
    # Save models (optional)
    try:
        import joblib
        joblib.dump(rf_model, "random_forest_model.pkl")
        mlp_model.save("mlp_model.h5")
        print("\n✅ Models saved: random_forest_model.pkl, mlp_model.h5")
    except Exception as e:
        print(f"\n⚠️  Could not save models: {e}")


if __name__ == "__main__":
    main()
