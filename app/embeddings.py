"""Local sentence embeddings.

Uses the ONNX build of ``all-MiniLM-L6-v2`` that ships with ChromaDB, run
through onnxruntime so the whole pipeline stays offline and free. If the model
is not present yet it is fetched once from Chroma's public model bucket.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_URL = f"https://chroma-onnx-models.s3.amazonaws.com/{MODEL_NAME}/onnx.tar.gz"
MODEL_DIR = Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME / "onnx"
EMBED_DIM = 384
MAX_TOKENS = 256

_lock = threading.Lock()
_session = None
_tokenizer = None


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


def _download_model() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = MODEL_DIR.parent / "onnx.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(MODEL_URL, archive)  # noqa: S310
    with tarfile.open(archive) as tar:
        tar.extractall(MODEL_DIR.parent)  # noqa: S202


def _load() -> None:
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return
    with _lock:
        if _session is not None and _tokenizer is not None:
            return
        if not MODEL_DIR.exists():
            try:
                _download_model()
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingUnavailable(
                    f"Could not download the embedding model: {exc}"
                ) from exc
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailable(f"onnxruntime/tokenizers missing: {exc}") from exc

        model_path = MODEL_DIR / "model.onnx"
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        if not model_path.exists() or not tokenizer_path.exists():
            raise EmbeddingUnavailable(
                f"Embedding model files missing under {MODEL_DIR}"
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        _session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(length=None, pad_id=0, pad_token="[PAD]")
        _tokenizer = tokenizer


def available() -> bool:
    try:
        _load()
        return True
    except Exception:  # noqa: BLE001
        return False


def embed(texts: list[str]) -> np.ndarray:
    """Embed ``texts`` into an L2-normalised ``(n, 384)`` float32 matrix."""

    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    _load()
    assert _session is not None and _tokenizer is not None

    encodings = [_tokenizer.encode(t if t.strip() else " ") for t in texts]
    max_len = max(len(e.ids) for e in encodings)
    pad_id = 0

    input_ids = np.full((len(texts), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
    token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)

    for row, enc in enumerate(encodings):
        length = len(enc.ids)
        input_ids[row, :length] = enc.ids
        attention_mask[row, :length] = enc.attention_mask
        token_type_ids[row, :length] = enc.type_ids

    outputs = _session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]

    mask = attention_mask.astype(np.float32)[:, :, None]
    summed = (outputs * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts

    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (pooled / norms).astype(np.float32)


def hash_fallback(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic bag-of-words hash vector, used only if ONNX is missing."""

    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec
