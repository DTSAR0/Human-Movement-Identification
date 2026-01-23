"""
MLP (Feedforward Neural Network) для класифікації активностей людини
на датасеті UCI HAR Dataset

Використовує 561 engineered features з файлів:
- train/X_train.txt, train/y_train.txt
- test/X_test.txt, test/y_test.txt
- activity_labels.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# ПАРАМЕТРИ (легко змінювати)
# ============================================================================

LEARNING_RATE = 1e-3
EPOCHS = 100
HIDDEN_NEURONS = 256  # Базовий розмір для Dense шарів (256/128/64)
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
DATASET_PATH = "human+activity+recognition+using+smartphones/UCI HAR Dataset"
RANDOM_STATE = 42  # Використовується для однакового split з Random Forest

# ============================================================================
# ІМПОРТИ
# ============================================================================

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
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
    print("Error: scikit-learn is not installed. Install it: pip install scikit-learn")
    exit(1)

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Error: TensorFlow is not installed. Install it: pip install tensorflow")
    exit(1)

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn is not installed. Will use matplotlib for plots.")


# ============================================================================
# ФУНКЦІЇ ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================================

def load_har_dataset(dataset_path):
    """
    Завантажує UCI HAR Dataset
    
    Args:
        dataset_path: шлях до папки 'UCI HAR Dataset'
    
    Returns:
        X_train, y_train, X_test, y_test, activity_names
    """
    dataset_path = Path(dataset_path)
    
    # Шляхи до файлів
    X_train_path = dataset_path / "train" / "X_train.txt"
    y_train_path = dataset_path / "train" / "y_train.txt"
    X_test_path = dataset_path / "test" / "X_test.txt"
    y_test_path = dataset_path / "test" / "y_test.txt"
    activity_labels_path = dataset_path / "activity_labels.txt"
    
    print("Loading UCI HAR dataset...")
    
    # Завантаження даних
    X_train = np.loadtxt(X_train_path)
    y_train = np.loadtxt(y_train_path, dtype=int)
    X_test = np.loadtxt(X_test_path)
    y_test = np.loadtxt(y_test_path, dtype=int)
    
    # Завантаження назв активностей
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
    
    # Конвертація міток з 1-6 в 0-5
    y_train = y_train - 1
    y_test = y_test - 1
    
    return X_train, y_train, X_test, y_test, activity_names


# ============================================================================
# ФУНКЦІЇ МОДЕЛІ
# ============================================================================

def build_mlp_model(input_shape, num_classes, hidden_neurons, learning_rate, dropout_rate):
    """
    Побудова MLP моделі
    
    Архітектура:
    Input(561) → Dense(256, ReLU) → BatchNorm → Dropout(0.3) 
    → Dense(128, ReLU) → BatchNorm → Dropout(0.3) 
    → Dense(64, ReLU) → Dropout(0.2) 
    → Dense(6, Softmax)
    
    Args:
        input_shape: кількість features (561)
        num_classes: кількість класів (6)
        hidden_neurons: базовий розмір для Dense шарів
        learning_rate: learning rate для Adam optimizer
        dropout_rate: dropout rate
    
    Returns:
        model: скомпільована Keras модель
    """
    # Розрахунок розмірів шарів на основі hidden_neurons
    layer1_size = hidden_neurons
    layer2_size = hidden_neurons // 2
    layer3_size = hidden_neurons // 4
    
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        
        # Перший Dense шар
        layers.Dense(layer1_size, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Другий Dense шар
        layers.Dense(layer2_size, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Третій Dense шар
        layers.Dense(layer3_size, activation='relu', name='dense_3'),
        layers.Dropout(dropout_rate * 0.67, name='dropout_3'),  # 0.2 при dropout_rate=0.3
        
        # Вихідний шар
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Компіляція моделі
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_mlp(model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose=1):
    """
    Навчання MLP моделі з EarlyStopping та ReduceLROnPlateau
    
    Args:
        model: скомпільована Keras модель
        X_train: тренувальні дані
        y_train: тренувальні мітки (one-hot encoded)
        X_val: валідаційні дані
        y_val: валідаційні мітки (one-hot encoded)
        epochs: максимальна кількість епох
        batch_size: розмір батчу
        verbose: verbosity mode
    
    Returns:
        history: історія навчання
    """
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
    
    # Навчання
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose
    )
    
    return history


# ============================================================================
# ФУНКЦІЇ ОЦІНКИ ТА ВІЗУАЛІЗАЦІЇ
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, activity_names):
    """
    Оцінка моделі та виведення метрик
    
    Args:
        model: навчена Keras модель
        X_train: тренувальні дані
        y_train: тренувальні мітки (one-hot encoded)
        X_test: тестові дані
        y_test: тестові мітки (one-hot encoded)
        activity_names: словник з назвами активностей
    
    Returns:
        metrics: словник з метриками
    """
    # Передбачення
    y_train_pred_proba = model.predict(X_train, verbose=0)
    y_test_pred_proba = model.predict(X_test, verbose=0)
    
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Обчислення метрик
    train_accuracy = accuracy_score(y_train_labels, y_train_pred)
    test_accuracy = accuracy_score(y_test_labels, y_test_pred)
    
    f1_macro = f1_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    f1_weighted = f1_score(y_test_labels, y_test_pred, average='weighted', zero_division='warn')
    precision_macro = precision_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    recall_macro = recall_score(y_test_labels, y_test_pred, average='macro', zero_division='warn')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_test_pred)
    
    # Classification Report
    num_classes = len(np.unique(y_test_labels))
    class_names = [activity_names.get(i+1, f"Class {i}") for i in range(num_classes)]  # type: ignore
    
    # Виведення результатів
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\nF1 Score (Macro):     {f1_macro:.4f}")
    print(f"F1 Score (Weighted):  {f1_weighted:.4f}")
    print(f"Precision (Macro):    {precision_macro:.4f}")
    print(f"Recall (Macro):       {recall_macro:.4f}")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    print(classification_report(y_test_labels, y_test_pred, target_names=class_names, zero_division='warn'))
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}\n")
    print(cm)
    
    # Збереження метрик
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return metrics


def plot_training_history(history):
    """
    Побудова графіків історії навчання
    
    Args:
        history: історія навчання з model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Побудова heatmap confusion matrix
    
    Args:
        cm: confusion matrix
        class_names: список назв класів
    """
    plt.figure(figsize=(10, 8))
    
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


# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    """
    Головна функція для навчання та оцінки MLP моделі
    """
    print("="*60)
    print("MLP CLASSIFIER FOR HUMAN ACTIVITY RECOGNITION")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Hidden neurons (base): {HIDDEN_NEURONS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Random state: {RANDOM_STATE}")
    print("="*60)
    
    # Перевірка залежностей
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is not installed!")
        return
    
    if not TENSORFLOW_AVAILABLE:
        print("Error: TensorFlow is not installed!")
        return
    
    # Встановлення seed для відтворюваності
    tf.keras.utils.set_random_seed(RANDOM_STATE)  # type: ignore

    
    # Завантаження даних
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at path: {dataset_path}")
        print("Please check the DATASET_PATH variable.")
        return
    
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, activity_names = load_har_dataset(dataset_path)
    
    # Використовуємо оригінальний train/test split (БЕЗ об'єднання - зберігаємо розділення людей)
    print(f"\n{'='*60}")
    print("USING ORIGINAL DATASET SPLIT (NO COMBINING)")
    print(f"{'='*60}")
    print("Train: Original UCI HAR train set")
    print("Test:  Original UCI HAR test set")
    print(f"Train: {X_train_orig.shape[0]} samples")  # type: ignore
    print(f"Test:  {X_test_orig.shape[0]} samples")  # type: ignore
    print("Note: Preserves original subject separation")
    
    # Нормалізація з StandardScaler (fit тільки на train!)
    print(f"\n{'='*60}")
    print("NORMALIZING FEATURES")
    print(f"{'='*60}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    print("✅ Features normalized (fit on train, transform on test)")
    
    # Validation split: 80% train / 20% val (з оригінального train set для early stopping MLP)
    print(f"\n{'='*60}")
    print("VALIDATION SPLIT: 80% train / 20% val (from original train set)")
    print(f"{'='*60}")
    print("Note: MLP splits original train into 80% train + 20% val for early stopping")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_orig,
        test_size=0.2,
        random_state=42,  # Однаковий seed для відтворюваності
        stratify=y_train_orig
    )
    print(f"Train (for MLP): {X_train_final.shape[0]} samples (80% of original train)")  # type: ignore
    print(f"Val (for MLP):   {X_val.shape[0]} samples (20% of original train)")  # type: ignore
    print(f"Test:            {X_test_scaled.shape[0]} samples (original test)")  # type: ignore
    
    # Конвертація міток в one-hot encoding
    num_classes = int(np.max(np.concatenate([y_train_orig, y_test_orig]))) + 1
    y_train_final_cat = to_categorical(y_train_final, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_orig, num_classes=num_classes)
    
    # Побудова моделі
    print(f"\n{'='*60}")
    print("BUILDING MLP MODEL")
    print(f"{'='*60}")
    model = build_mlp_model(
        input_shape=X_train_final.shape[1],  # type: ignore
        num_classes=num_classes,
        hidden_neurons=HIDDEN_NEURONS,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Навчання моделі
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    history = train_mlp(
        model,
        X_train_final, y_train_final_cat,
        X_val, y_val_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Візуалізація історії навчання
    print(f"\n{'='*60}")
    print("TRAINING HISTORY")
    print(f"{'='*60}")
    plot_training_history(history)
    
    # Оцінка моделі
    metrics = evaluate_model(
        model,
        X_train_final, y_train_final_cat,
        X_test_scaled, y_test_cat,
        activity_names
    )
    
    # Візуалізація confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX VISUALIZATION")
    print(f"{'='*60}")
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['class_names'])
    
    # Збереження моделі
    model_path = "mlp_har_model.h5"
    model.save(model_path)
    print(f"\n✅ Model saved as '{model_path}'")
    
    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATION COMPLETED!")
    print(f"{'='*60}\n")
    
    return model, metrics, history


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()

