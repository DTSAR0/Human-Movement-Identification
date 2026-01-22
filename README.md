# Human-Movement-Identification

Проєкт для розпізнавання руху людини (Human Activity Recognition) з використанням датасету UCI HAR.

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

## Використання

### Random Forest класифікатор

Для навчання та валідації Random Forest класифікатора:

```bash
python Human_movement_identification.py random_forest
```

Або в Python коді:

```python
from Human_movement_identification import main_random_forest

model, metrics = main_random_forest()
```

### Метрики валідації

Класифікатор автоматично обчислює та виводить:
- **Accuracy** (Точність) - загальна точність класифікації
- **F1 Score** - макро-середнє та зважене середнє, а також по кожному класу
- **Precision** (Точність) - макро-середнє
- **Recall** (Повнота) - макро-середнє
- **Confusion Matrix** (Матриця плутанини) - візуалізація помилок класифікації

---

