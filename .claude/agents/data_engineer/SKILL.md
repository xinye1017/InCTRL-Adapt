# Data Engineer Skills

## 1. Synthetic Anomaly Generation
When tasked with generating anomalies:
1. Analyze the normal background texture (e.g., structural vs. non-structural).
2. Select appropriate synthesis method:
   - For textures (carpet, wood): Use Perlin noise-based blending.
   - For objects (bottle, cable): Use localized Cut-Paste with seamless cloning.
3. Ensure generated masks are pixel-perfect and aligned with the visual defect.

## 2. Industrial Augmentation Pipeline
When building data pipelines:
1. Implement color jittering that respects lighting variations in factory settings.
2. Add rotation/flip augmentations ONLY if the object lacks strict orientation (e.g., screw = yes, transistor = no).
3. Output a PyTorch `transforms.Compose` pipeline optimized for speed.

## 3. Imbalance Handling
When addressing anomaly scarcity:
1. Implement oversampling strategies for the anomalous class.
2. Design few-shot data loading mechanisms (k-shot sampling).
3. Validate that validation/test sets maintain their original, real-world distribution.
