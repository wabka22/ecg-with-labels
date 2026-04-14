import numpy as np
import matplotlib.pyplot as plt


def plot_signal_with_labels(signal_path, labels_path, channel=0):
    signal = np.load(signal_path)
    labels = np.load(labels_path)

    sig = signal[channel]
    lab = labels[channel]

    plt.figure(figsize=(15, 5))
    plt.plot(sig, label="ECG signal", linewidth=1)

    unique_labels = np.unique(lab)
    unique_labels = unique_labels[unique_labels != -1]

    colors = {
        0: "green",
        1: "red",
        2: "blue",
        3: "orange"
    }

    for label in unique_labels:
        mask = lab == label

        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        start = indices[0]

        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                end = indices[i - 1]
                plt.axvspan(start, end, alpha=0.3,
                            color=colors.get(label, "gray"),
                            label=f"label {label}")
                start = indices[i]

        plt.axvspan(start, indices[-1], alpha=0.3,
                    color=colors.get(label, "gray"))

    plt.title(f"Channel {channel}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    plot_signal_with_labels(
        signal_path="output/0/signal.npy",
        labels_path="output/0/labels.npy",
        channel=0
    )