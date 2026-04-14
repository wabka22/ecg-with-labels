# ECG Signal Processor

Модуль для подготовки ECG-данных к обучению моделей.

## Входные данные

- `*.npy` — сигнал (shape: channels x samples)
- `*.json` — разметка (сегменты)

Файлы должны совпадать по имени:
0.npy ↔ 0.json
1.npy ↔ 1.json

## Использование

### Загрузка одного примера

```python
from ecg_signal_processor import load_sample

signal, labels = load_sample(
    "data/ecs_short/0.npy",
    "data/markings/0.json"
)
```

### Dataset для обучения

```python
from ecg_signal_processor import ECGDataset

dataset = ECGDataset(
    signal_dir="data/ecs_short",
    markup_dir="data/markings"
)

x, y = dataset[0]
print(x.shape, y.shape)
```

### DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for x, y in loader:
    print(x.shape, y.shape)
    break
```

### Визуализация

```python
from ecg_signal_processor import plot_signal_with_labels

plot_signal_with_labels(
    "output/0/signal.npy",
    "output/0/labels.npy",
    channel=0
)
```

## Формат данных

- `signal` — вход (ECG)
- `labels` — разметка

shape: (channels, samples)

Значения:
- -1 — нет разметки
- 0,1,2... — классы
