import torch
from torch import nn
import numpy as np
from pathlib import Path
from torchvision import datasets
import torch.nn.functional as F
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).resolve().parent / "data"

th = torch.rand(5,3)
print(f"Shape of tensor: {th.shape}")
print(f"Datatype of tensor: {th.dtype}")
print(f"Device tensor is stored on: {th.device}")

training_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Build the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training.
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# With target_transforms
# ds = datasets.FashionMNIST(
#     root=DATA_DIR,
#     train=True,
#     download=True,
#     transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
#     target_transform=v2.Lambda(
#         lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
#     ),
# )

# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")

# plt.show()

# if torch.cuda.is_available():
#     print(f"Torch is running on {torch.cuda.get_device_name(0)}")
# else:
#     print("Torch is not running on any accelerator")

# pytorch tensors: encode inputs and outputs of a model, as well as the model's parameters

# To Learn
# Arithmetic operations on tensors

# Dataset: stores the samples and their labels
# DataLoader: wraps an iterable around the Dataset to enable easy access to batches of data