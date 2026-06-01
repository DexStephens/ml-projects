import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

input_image = torch.rand(3,28,28)
print(input_image.size())

# Change 2d array of 28x28 to 1d array of 784, why? A linear layer expects a 1D array that it does multiplication on. Convolutional layers expect a 3D array (width, height, depth/channels).
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# import torch
# import matplotlib.pyplot as plt
# from torchvision import datasets
# from torchvision.transforms import v2

# # 1) Raw: no transform → PIL Image
# raw_ds = datasets.FashionMNIST(
#     root="data", train=True, download=True, transform=None
# )
# pil_img, label = raw_ds[0]

# # 2) Build pipelines you can apply to that PIL image
# only_to_image = v2.Compose([v2.ToImage()])
# full_pipeline = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
# ])

# t_after_image = only_to_image(pil_img)
# t_final = full_pipeline(pil_img)

# def describe(name, x):
#   if hasattr(x, "size"):  # PIL
#     print(f"{name}: type=PIL, size={x.size}, mode={x.mode}")
#   else:
#     print(
#       f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
#       f"min={x.min().item():.4f}, max={x.max().item():.4f}"
#     )

# describe("0) Raw PIL", pil_img)
# describe("1) After ToImage", t_after_image)
# describe("2) After ToImage + ToDtype(scale=True)", t_final)

# # 3) Side-by-side plot (they should look almost identical visually)
# fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# axes[0].imshow(pil_img, cmap="gray")
# axes[0].set_title("Raw PIL\nuint8, 0–255")
# axes[0].axis("off")

# axes[1].imshow(t_after_image.squeeze(), cmap="gray")
# axes[1].set_title(f"After ToImage\n{t_after_image.dtype}")
# axes[1].axis("off")

# axes[2].imshow(t_final.squeeze(), cmap="gray")
# axes[2].set_title(f"After ToDtype + scale\n{t_final.dtype}")
# axes[2].axis("off")

# plt.suptitle(f"label={label}")
# plt.tight_layout()
# plt.show()