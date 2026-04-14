from ecg_signal_processor import ECGDataset

dataset = ECGDataset(
    signal_dir="data/ecs_short",
    markup_dir="data/markings"
)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("X shape:", x.shape)
print("Y shape:", y.shape)