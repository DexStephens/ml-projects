# Computer Vision Learning

This repo is for learning computer vision, PyTorch, neural networks, and model deployment through small examples and project plans.

## Repository layout

```text
computer-vision/
├── learnings/              # notes + small PyTorch experiments
│   ├── README.md
│   ├── autograd_and_requires_grad.py
│   ├── model_layers.py
│   ├── transform_pipeline_demo.py
│   └── assets/
├── projects/
│   ├── fashion_mnist/    # FashionMNIST classifier walkthrough
│   └── security_camera/  # Raspberry Pi security camera plan
├── pyproject.toml
└── uv.lock
```

## Projects

| Project | Path | Description |
|---------|------|-------------|
| FashionMNIST Classifier | [`projects/fashion_mnist/`](projects/fashion_mnist/) | Datasets, transforms, dataloaders, model layers, inference |
| AI Security Camera | [`projects/security_camera/`](projects/security_camera/) | End-to-end Pi + inference server architecture (plan) |

## Learnings

Concept notes and scratch scripts live in [`learnings/`](learnings/):

- Cross entropy vs BCE
- Linear vs convolutional layers
- Backpropagation and `requires_grad`
- Layer and transform experiments

## Environment setup

This project uses `uv` and a local virtual environment. The `.venv` directory should stay local and should not be committed.

```bash
uv sync
source .venv/bin/activate
```

The VS Code/Cursor debugger should point to the local interpreter:

```text
${workspaceFolder}/.venv/bin/python
```

Generated datasets, model checkpoints, caches, and local environment files should stay out of Git. See `.gitignore` for the current rules.
