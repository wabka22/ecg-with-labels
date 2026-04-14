import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_signal(signal_path: Path) -> np.ndarray:
    """
    Загружает ECG сигнал из .npy.

    Если файл повреждён лишними байтами перед заголовком NPY,
    пытается восстановить его по сигнатуре NUMPY.
    """
    try:
        return np.load(signal_path, allow_pickle=False)
    except Exception:
        pass

    raw = signal_path.read_bytes()
    marker = b"NUMPY"
    pos = raw.find(marker)

    if pos == -1:
        raise ValueError(
            f"Не удалось загрузить сигнал и не найден заголовок NPY: {signal_path}"
        )

    recovered = b"\x93" + raw[pos:]
    tmp_path = signal_path.with_suffix(".recovered.npy")
    tmp_path.write_bytes(recovered)

    return np.load(tmp_path, allow_pickle=False)


def build_label_matrix(signal: np.ndarray, markup: dict, background_value: int) -> np.ndarray:
    """
    Создаёт матрицу меток формы [channels, samples].

    background_value:
        значение для точек без разметки
    """
    if signal.ndim != 2:
        raise ValueError(f"Ожидался сигнал формы [channels, samples], получено: {signal.shape}")

    n_channels, n_samples = signal.shape
    labels = np.full((n_channels, n_samples), background_value, dtype=np.int32)

    segments_by_channel = markup.get("Segments")
    if segments_by_channel is None:
        raise ValueError("В JSON отсутствует ключ 'Segments'")

    if len(segments_by_channel) != n_channels:
        print(
            f"Предупреждение: число каналов в сигнале ({n_channels}) "
            f"не совпадает с числом каналов в разметке ({len(segments_by_channel)})"
        )

    for ch_idx, channel_segments in enumerate(segments_by_channel):
        if ch_idx >= n_channels:
            break

        for seg in channel_segments:
            seg_type = int(seg["Type"])
            start = max(0, int(seg["StartMark"]))
            end = min(n_samples - 1, int(seg["EndMark"]))

            if start <= end:
                labels[ch_idx, start:end + 1] = seg_type

    return labels


def build_dataframe(signal: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Преобразует сигнал и разметку в таблицу:
    sample | ch0_signal | ch0_label | ch1_signal | ch1_label | ...
    """
    n_channels, n_samples = signal.shape
    data = {"sample": np.arange(n_samples)}

    for ch in range(n_channels):
        data[f"ch{ch}_signal"] = signal[ch]
        data[f"ch{ch}_label"] = labels[ch]

    return pd.DataFrame(data)


def process_file(signal_path: Path, markup_path: Path, output_dir: Path, background_value: int) -> None:
    """
    Обрабатывает один пример и сохраняет:
    - signal.npy
    - labels.npy
    - data.csv
    """
    signal = load_signal(signal_path)

    with open(markup_path, "r", encoding="utf-8") as f:
        markup = json.load(f)

    labels = build_label_matrix(signal, markup, background_value)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "signal.npy", signal)
    np.save(output_dir / "labels.npy", labels)

    df = build_dataframe(signal, labels)
    df.to_csv(output_dir / "data.csv", index=False)

    print(f"Обработан файл: {signal_path.name} -> {output_dir}")


def main():
    config = load_config("config.json")

    signal_dir = Path(config["input"]["signal_dir"])
    markup_dir = Path(config["input"]["markup_dir"])
    output_base_dir = Path(config["output"]["base_dir"])
    background_value = int(config["params"].get("background_value", -1))

    if not signal_dir.exists():
        raise FileNotFoundError(f"Папка с сигналами не найдена: {signal_dir}")

    if not markup_dir.exists():
        raise FileNotFoundError(f"Папка с разметкой не найдена: {markup_dir}")

    signal_files = sorted(signal_dir.glob("*.npy"))

    if not signal_files:
        print(f"В папке нет .npy файлов: {signal_dir}")
        return

    processed_count = 0
    skipped_count = 0

    for signal_path in signal_files:
        file_id = signal_path.stem
        markup_path = markup_dir / f"{file_id}.json"

        if not markup_path.exists():
            print(f"Нет разметки для файла {file_id}, пропуск")
            skipped_count += 1
            continue

        output_dir = output_base_dir / file_id
        process_file(signal_path, markup_path, output_dir, background_value)
        processed_count += 1

    print("\nГотово")
    print(f"Обработано: {processed_count}")
    print(f"Пропущено: {skipped_count}")


if __name__ == "__main__":
    main()