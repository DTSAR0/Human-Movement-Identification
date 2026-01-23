# Налаштування TensorFlow для проекту

## Проблема
Cursor/IDE показує помилку імпорту TensorFlow, навіть якщо він встановлений.

## Рішення

### Варіант 1: Використати Python 3.12 (рекомендовано)
TensorFlow не підтримує Python 3.14. Створіть нове venv з Python 3.12:

```bash
# Встановіть Python 3.12 (якщо немає)
# brew install python@3.12

# Створіть нове venv з Python 3.12
python3.12 -m venv venv

# Активуйте venv
source venv/bin/activate

# Встановіть залежності
pip install -r requirements.txt
pip install tensorflow
```

### Варіант 2: Налаштувати IDE для використання правильного Python

1. Відкрийте Command Palette (`Cmd+Shift+P`)
2. Виберіть "Python: Select Interpreter"
3. Виберіть Python, де встановлений TensorFlow
   - Або виберіть `./venv/bin/python` (якщо TensorFlow встановлений там)
   - Або системний Python (якщо TensorFlow встановлений глобально)

### Варіант 3: Встановити TensorFlow в поточне середовище

Якщо використовуєте Python 3.12 або нижче:

```bash
source venv/bin/activate
pip install tensorflow
```

## Перевірка

```bash
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

Якщо виводить версію - все працює!
