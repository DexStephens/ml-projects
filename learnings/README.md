# Learnings

Notes and small scripts from working through PyTorch and computer vision fundamentals.

## Scripts

| File | Topic |
|------|--------|
| `autograd_and_requires_grad.py` | `grad_fn`, `loss.backward()`, and `requires_grad` |
| `model_layers.py` | `Flatten`, `Linear`, `ReLU`, `Sequential`, `Softmax` |
| `transform_pipeline_demo.py` | FashionMNIST transform steps (commented walkthrough) |

## Assets

- `assets/computational_graph_basic_diagram.png` — autograd / computation graph reference

## Cross Entropy vs BCE

Cross entropy is for mutually exclusive classes, where each example belongs to exactly one class.

Example:

- Each image is a T-shirt, pants, or shoes.

BCE is for independent yes/no labels, where each example can have multiple labels at the same time.

Example:

- Each image can have person: yes/no, car: yes/no, dog: yes/no.

For FashionMNIST, `CrossEntropyLoss` is the usual choice because each image has exactly one label.

## Linear vs Convolutional Layers

Use **linear layers** when the input is a fixed feature vector, such as tabular data or embeddings. Each input position has a stable meaning, like age, income, score, or an extracted feature. Linear layers are also commonly used at the end of neural networks to map learned features to final class scores.

Use **convolutional layers** when the input has spatial or local structure, such as images, video frames, spectrograms, or some time series. Convolutions preserve the idea that nearby values are related and can detect local patterns like edges, textures, and shapes.

For images, a simple linear model usually flattens `[channels, height, width]` into one long vector, which loses the explicit row/column/channel structure. A convolutional model keeps the image-like shape longer and learns visual features before flattening near the end.

Rule of thumb:

- **Linear / MLP**: fixed feature vectors, tabular data, embeddings, final classifier heads.
- **Convolutional**: grid-like data where neighboring values matter, especially images.
- **Transformers**: sequences, text, long-range relationships, and increasingly image models.
- **Tree models**: strong baseline for many traditional tabular ML problems.

## Backpropagation

Backpropagation computes how much each model parameter contributed to the loss. Parameters are then adjusted according to the gradient of the loss function with respect to each parameter. PyTorch uses `torch.autograd` for this.

## `requires_grad`

`requires_grad=True` tells PyTorch to track operations on a tensor so it can compute gradients during backpropagation.

Model parameters usually need gradients because training updates them:

```python
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()

print(w.grad)
print(b.grad)
```

After `loss.backward()`, PyTorch fills in `.grad` for tensors that require gradients. The optimizer later uses those gradients to update the model weights. Input data usually does not need `requires_grad=True`; the model's weights are what we want to learn.

For production inference, use `model.eval()` plus `torch.no_grad()` or `torch.inference_mode()` so PyTorch does not build a gradient graph for each prediction.
