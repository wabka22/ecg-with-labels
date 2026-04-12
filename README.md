# ECG Preprocessor

A script for combining ECG signal and markup into a convenient format for analysis and ML.

## Входные данные

- `.npy` — signal (shape: channels x samples)
- `.json` — segment marking

## Настройка

All paths are specified in `config.json`:

```json
{
  "input": {
    "signal_path": "data/0.npy",
    "markup_path": "data/markup.json"
  },
  "output": {
    "labels_path": "output/labels.npy",
    "csv_path": "output/ecg_with_markup.csv"
  }
}