# Model Engineer Skills

## 1. Architecture Modification Protocol
When adding new modules to InCTRL:
1. Analyze the feature map dimensions at the insertion point.
2. Design the module (e.g., cross-attention, adapter) to be zero-initialized or identity-mapping at start to preserve pre-trained knowledge.
3. Provide the exact PyTorch `nn.Module` code with typing and shape documentation in comments.

## 2. Prompt Tuning Implementation
When tasked with prompt optimization:
1. Implement learnable prompt embeddings (e.g., Context Optimization - CoOp).
2. Ensure prompt dimensions match the CLIP text encoder expectations.
3. Design the integration so text and visual embeddings remain aligned in the joint space.

## 3. Model Surgery
When extracting or bypassing layers:
1. Use PyTorch forward hooks (`register_forward_hook`) rather than rewriting large third-party classes.
2. Verify gradient flow is maintained (or intentionally stopped via `requires_grad=False`).
3. Output an architecture diagram (in text) explaining the change.
