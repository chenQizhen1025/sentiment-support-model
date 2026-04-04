# Local Model Placeholder

This directory is intentionally kept as a lightweight placeholder.

Model weights are not stored in Git. Download the model on the target machine before training or inference.

Recommended options:

- Use `scripts/training/download_qwen_model.py` after updating the environment as needed.
- Or download directly with ModelScope to this directory:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='models/Qwen/Qwen3-4B-Instruct-2507')"
```
