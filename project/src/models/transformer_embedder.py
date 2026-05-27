from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from src.config import PROJECT_ROOT


def resolve_model_source(model_name: str) -> tuple[str, bool]:
    path = Path(model_name)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return str(path), True
    return model_name, False


@lru_cache(maxsize=2)
def load_transformer(
    model_name: str,
    cache_dir: str | None = None,
    device_preference: str = "auto",
):
    resolved_model_name, is_local_path = resolve_model_source(model_name)
    if cache_dir is not None:
        os.environ.setdefault("HF_HOME", str(os.path.dirname(cache_dir)))
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        cache_dir=cache_dir,
        local_files_only=is_local_path or cache_dir is not None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "TRANSFORMER_DEVICE=cuda is configured, but PyTorch cannot access CUDA. "
            "Run training outside the sandbox or start Docker with GPU access."
        )
    device = "cuda" if device_preference in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        resolved_model_name,
        cache_dir=cache_dir,
        local_files_only=is_local_path or cache_dir is not None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    model_name: str,
    cache_dir: str | None,
    device_preference: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    tokenizer, model, device = load_transformer(model_name, cache_dir, device_preference)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.float().cpu().numpy())
    return np.vstack(embeddings)
