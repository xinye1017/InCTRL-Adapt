# Training Engineer Skills

## 1. Loss Function Design
When formulating loss for anomaly detection:
1. Balance classification (image-level) and segmentation (pixel-level) objectives.
2. Use dynamic weighting or uncertainty weighting if losses operate on different scales.
3. Provide the PyTorch loss implementation, ensuring numerical stability (e.g., adding `eps` to denominators).

## 2. Optimizer & Scheduler Configuration
When setting up training dynamics:
1. Select optimizers robust to noisy gradients (e.g., AdamW over standard SGD for transformers).
2. Implement linear warmup followed by cosine annealing for learning rates.
3. Provide exact hyperparameter recommendations (LR, weight decay, betas) based on batch size.

## 3. Training Loop Implementation
When writing the training loop:
1. Integrate `torch.cuda.amp` for mixed precision training.
2. Ensure `model.train()` and `model.eval()` are correctly toggled.
3. Implement logic to freeze specific layers (e.g., freezing CLIP vision encoder while training adapters).
