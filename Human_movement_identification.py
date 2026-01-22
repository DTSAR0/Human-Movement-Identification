import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Перевірка наявності бібліотек для нейронної мережі
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Увага: TensorFlow не встановлено. Для навчання нейронної мережі встановіть: pip install tensorflow")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Увага: scikit-learn не встановлено. Встановіть: pip install scikit-learn")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Увага: seaborn не встановлено. Встановіть: pip install seaborn")

def display_images_from_folder(folder_path, num_images=10):
    """
    Відображає перші N фото з вказаної папки
    
    Args:
        folder_path: шлях до папки з фото
        num_images: кількість фото для відображення (за замовчуванням 10)
    """
    # Підтримувані формати зображень
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Перевірка чи існує папка
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Помилка: Папка '{folder_path}' не існує!")
        return
    
    # Збір всіх файлів зображень
    image_files = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print(f"У папці '{folder_path}' не знайдено зображень.")
        return False
    
    # Обмеження кількості зображень
    num_images = min(num_images, len(image_files))
    image_files = image_files[:num_images]
    
    # Створення сітки для відображення
    cols = 5  # Кількість стовпців
    rows = (num_images + cols - 1) // cols  # Кількість рядків (округлення вгору)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle(f'Перші {num_images} фото з папки "{folder.name}"', fontsize=16, fontweight='bold')
    
    # Якщо тільки одне зображення
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Відображення кожного зображення
    for idx, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{img_path.name}', fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            print(f"Помилка при завантаженні {img_path.name}: {e}")
            axes[idx].text(0.5, 0.5, 'Помилка\nзавантаження', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Приховування зайвих вікон
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Успішно відображено {num_images} зображень з {len(image_files)} знайдених.")
    return True


def display_har_dataset_signals(dataset_path, num_samples=10):
    """
    Відображає сигнали з датасету UCI HAR (Human Activity Recognition)
    
    Args:
        dataset_path: шлях до папки з датасетом
        num_samples: кількість зразків для відображення
    """
    dataset_path = Path(dataset_path)
    train_path = dataset_path / "train" / "Inertial Signals"
    test_path = dataset_path / "test" / "Inertial Signals"
    
    # Вибираємо тестовий набір
    if not test_path.exists():
        print(f"Помилка: Не знайдено папку з тестовими даними: {test_path}")
        return
    
    # Завантаження сигналів
    signals = {}
    signal_names = {
        'body_acc_x': 'Body Acceleration X',
        'body_acc_y': 'Body Acceleration Y',
        'body_acc_z': 'Body Acceleration Z',
        'body_gyro_x': 'Body Gyroscope X',
        'body_gyro_y': 'Body Gyroscope Y',
        'body_gyro_z': 'Body Gyroscope Z',
        'total_acc_x': 'Total Acceleration X',
        'total_acc_y': 'Total Acceleration Y',
        'total_acc_z': 'Total Acceleration Z'
    }
    
    for signal_key, signal_label in signal_names.items():
        file_path = test_path / f"{signal_key}_test.txt"
        if file_path.exists():
            signals[signal_key] = np.loadtxt(file_path)
            print(f"Завантажено {signal_key}: {signals[signal_key].shape}")
    
    if not signals:
        print("Помилка: Не знайдено файлів з сигналами!")
        return
    
    # Завантаження міток активностей
    labels_path = dataset_path / "test" / "y_test.txt"
    activity_labels_path = dataset_path / "activity_labels.txt"
    
    labels = None
    activity_names = {}
    
    if labels_path.exists():
        labels = np.loadtxt(labels_path, dtype=int)
        print(f"Завантажено мітки: {labels.shape}")
    
    if activity_labels_path.exists():
        with open(activity_labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    activity_names[int(parts[0])] = ' '.join(parts[1:])
        print(f"Завантажено назви активностей: {activity_names}")
    
    # Обмеження кількості зразків
    num_samples = min(num_samples, len(list(signals.values())[0]))
    
    # Створення графіків
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 2.5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Перші {num_samples} зразків сигналів руху з датасету UCI HAR', 
                 fontsize=16, fontweight='bold')
    
    # Вибір сигналів для відображення (прискорення тіла по трьох осях)
    selected_signals = ['body_acc_x', 'body_acc_y', 'body_acc_z']
    
    for sample_idx in range(num_samples):
        for col_idx, signal_key in enumerate(selected_signals):
            if signal_key in signals:
                signal_data = signals[signal_key][sample_idx]
                time_steps = np.arange(len(signal_data))
                
                axes[sample_idx, col_idx].plot(time_steps, signal_data, linewidth=1.5)
                axes[sample_idx, col_idx].set_title(f'{signal_names[signal_key]}', fontsize=10)
                axes[sample_idx, col_idx].set_xlabel('Час (відліки)', fontsize=9)
                axes[sample_idx, col_idx].set_ylabel('Значення', fontsize=9)
                axes[sample_idx, col_idx].grid(True, alpha=0.3)
                
                # Додавання мітки активності
                if labels is not None and sample_idx < len(labels):
                    activity_id = labels[sample_idx]
                    activity_name = activity_names.get(activity_id, f"Activity {activity_id}")
                    axes[sample_idx, col_idx].text(0.02, 0.98, f'Зразок {sample_idx+1}: {activity_name}',
                                                   transform=axes[sample_idx, col_idx].transAxes,
                                                   fontsize=8, verticalalignment='top',
                                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nУспішно відображено {num_samples} зразків сигналів руху.")


def load_har_dataset(dataset_path):
    """
    Завантажує датасет UCI HAR для навчання нейронної мережі
    
    Args:
        dataset_path: шлях до папки 'UCI HAR Dataset'
    
    Returns:
        X_train, y_train, X_test, y_test, activity_names
    """
    dataset_path = Path(dataset_path)
    
    # Завантаження тренувальних даних
    X_train_path = dataset_path / "train" / "X_train.txt"
    y_train_path = dataset_path / "train" / "y_train.txt"
    
    # Завантаження тестових даних
    X_test_path = dataset_path / "test" / "X_test.txt"
    y_test_path = dataset_path / "test" / "y_test.txt"
    
    # Завантаження назв активностей
    activity_labels_path = dataset_path / "activity_labels.txt"
    
    print("Завантаження датасету UCI HAR...")
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
    print(f"y_train: {y_train.shape}, унікальні класи: {np.unique(y_train)}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}, унікальні класи: {np.unique(y_test)}")
    
    # Перетворення міток з 1-6 на 0-5 для сумісності з TensorFlow
    y_train = y_train - 1
    y_test = y_test - 1
    
    return X_train, y_train, X_test, y_test, activity_names


def build_neural_network(input_shape=561, num_classes=6):
    """
    Створює нейронну мережу для розпізнавання руху людини
    
    Args:
        input_shape: розмірність вхідних даних (561 ознака)
        num_classes: кількість класів (6 активностей)
    
    Returns:
        model: скомпільована модель Keras
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow не встановлено. Встановіть: pip install tensorflow")
    
    model = keras.Sequential([
        # Вхідний шар
        layers.Dense(512, activation='relu', input_shape=(input_shape,), name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.5, name='dropout_1'),
        
        # Приховані шари
        layers.Dense(256, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.4, name='dropout_2'),
        
        layers.Dense(128, activation='relu', name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Dropout(0.3, name='dropout_3'),
        
        layers.Dense(64, activation='relu', name='dense_4'),
        layers.Dropout(0.2, name='dropout_4'),
        
        # Вихідний шар
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Компіляція моделі
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Навчає модель нейронної мережі
    
    Args:
        model: модель Keras
        X_train, y_train: тренувальні дані
        X_val, y_val: валідаційні дані
        epochs: кількість епох
        batch_size: розмір батчу
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow не встановлено.")
    
    # Callbacks для покращення навчання
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Навчання
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, activity_names):
    """
    Оцінює модель на тестових даних та відображає результати
    
    Args:
        model: навчена модель
        X_test, y_test: тестові дані
        activity_names: словник з назвами активностей
    """
    # Передбачення
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Точність
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"Результати на тестових даних:")
    print(f"Точність (Accuracy): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Втрати (Loss): {test_loss:.4f}")
    print(f"{'='*60}\n")
    
    # Classification report
    if SKLEARN_AVAILABLE:
        class_names = [activity_names.get(i+1, f"Class {i}") for i in range(len(activity_names))]
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        else:
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.colorbar()
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.yticks(range(len(class_names)), class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.title('Матриця плутанини (Confusion Matrix)')
    plt.ylabel('Справжні мітки')
    plt.xlabel('Передбачені мітки')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Відображає графіки процесу навчання
    
    Args:
        history: історія навчання моделі
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Графік точності
    axes[0].plot(history.history['accuracy'], label='Тренувальна точність', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Валідаційна точність', linewidth=2)
    axes[0].set_xlabel('Епоха')
    axes[0].set_ylabel('Точність')
    axes[0].set_title('Точність моделі')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Графік втрат
    axes[1].plot(history.history['loss'], label='Тренувальні втрати', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Валідаційні втрати', linewidth=2)
    axes[1].set_xlabel('Епоха')
    axes[1].set_ylabel('Втрати')
    axes[1].set_title('Втрати моделі')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def check_and_display_data(folder_path):
    """
    Перевіряє папку та відображає дані залежно від їх типу
    
    Args:
        folder_path: шлях до папки для перевірки
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Помилка: Папка '{folder_path}' не існує!")
        return
    
    print(f"\nПеревірка папки: {folder_path}")
    print("=" * 60)
    
    # Перевірка на наявність фото
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    # Перевірка на наявність датасету UCI HAR
    har_dataset_path = None
    if (folder / "UCI HAR Dataset").exists():
        har_dataset_path = folder / "UCI HAR Dataset"
    elif (folder / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset").exists():
        har_dataset_path = folder / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
    
    print(f"\nЗнайдено:")
    print(f"  - Зображень: {len(image_files)}")
    print(f"  - Датасет UCI HAR: {'Так' if har_dataset_path else 'Ні'}")
    
    # Відображення фото якщо вони є
    if image_files:
        print(f"\nВідображення фото...")
        display_images_from_folder(folder_path, num_images=10)
    elif har_dataset_path:
        print(f"\nВідображення сигналів з датасету UCI HAR...")
        display_har_dataset_signals(har_dataset_path, num_samples=10)
    else:
        print(f"\nУ папці не знайдено ні фото, ні датасету UCI HAR.")
        print(f"Перевірте шлях до папки з даними.")


def main_train_neural_network():
    """
    Головна функція для навчання нейронної мережі на датасеті UCI HAR
    """
    if not TENSORFLOW_AVAILABLE:
        print("Помилка: TensorFlow не встановлено!")
        print("Встановіть за допомогою: pip install tensorflow")
        return
    
    if not SKLEARN_AVAILABLE:
        print("Помилка: scikit-learn не встановлено!")
        print("Встановіть за допомогою: pip install scikit-learn")
        return
    
    # Шлях до датасету
    dataset_path = Path("human+activity+recognition+using+smartphones/UCI HAR Dataset")
    
    if not dataset_path.exists():
        print(f"Помилка: Датасет не знайдено за шляхом: {dataset_path}")
        print("Перевірте шлях до папки з датасетом!")
        return
    
    # Завантаження датасету
    X_train, y_train, X_test, y_test, activity_names = load_har_dataset(dataset_path)
    
    # Розділення тренувальних даних на train та validation
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn не встановлено. Встановіть: pip install scikit-learn")
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nРозділення даних:")
    print(f"Тренувальні: {X_train_split.shape[0]} зразків")
    print(f"Валідаційні: {X_val.shape[0]} зразків")
    print(f"Тестові: {X_test.shape[0]} зразків")
    
    # Створення моделі
    print(f"\n{'='*60}")
    print("Створення нейронної мережі...")
    print(f"{'='*60}")
    model = build_neural_network(input_shape=561, num_classes=6)
    model.summary()
    
    # Навчання моделі
    print(f"\n{'='*60}")
    print("Початок навчання моделі...")
    print(f"{'='*60}")
    history = train_model(
        model, 
        X_train_split, y_train_split, 
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Візуалізація історії навчання
    plot_training_history(history)
    
    # Оцінка моделі
    evaluate_model(model, X_test, y_test, activity_names)
    
    # Збереження моделі
    model.save('har_neural_network.h5')
    print(f"\nМодель збережено як 'har_neural_network.h5'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Запуск навчання нейронної мережі
        main_train_neural_network()
    else:
        # Перевірка та відображення даних
        data_folder = "."
        check_and_display_data(data_folder)
