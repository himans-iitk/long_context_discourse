"""Linear probing of transformer hidden states (Experiment 3).

Heavy imports (``torch``, ``transformers``) are deferred to construction
time so the rest of the package can be imported on machines without a GPU
or without the ``probe`` extras installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class LayerProbeResult:
    layer: int
    mean_acc: float
    std_acc: float
    max_acc: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "layer": self.layer,
            "mean_acc": self.mean_acc,
            "std_acc": self.std_acc,
            "max_acc": self.max_acc,
        }


@dataclass(frozen=True)
class PositionProbeResult:
    bucket: int
    position_label: str
    mean_acc: float
    std_acc: float
    n_spans: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "bucket": self.bucket,
            "position_label": self.position_label,
            "mean_acc": self.mean_acc,
            "std_acc": self.std_acc,
            "n_spans": self.n_spans,
        }


class HiddenStateExtractor:
    """Wraps a HuggingFace causal LM exposing hidden states."""

    def __init__(
        self,
        model_name: str,
        *,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        max_length: int = 128,
    ) -> None:
        try:
            import torch  # noqa: F401  (imported here to keep optional)
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "Experiment 3 needs torch+transformers. Install the 'probe' extras: "
                "pip install -e .[probe]"
            ) from exc

        kwargs: dict[str, Any] = {"output_hidden_states": True, "device_map": device_map}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch as _torch

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:  # pragma: no cover - depends on platform
                # bitsandbytes not available (e.g. Apple Silicon); fall back.
                kwargs.pop("quantization_config", None)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(model_name, **kwargs)
        self._model.eval()
        self._max_length = max_length
        self._n_layers = self._model.config.num_hidden_layers + 1  # +1 = embeddings

    @property
    def num_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_size(self) -> int:
        return int(self._model.config.hidden_size)

    def embed_span(self, text: str, *, layer_idx: int = -1) -> np.ndarray:
        """Mean-pooled hidden state at ``layer_idx`` (BOS excluded)."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden]
        # exclude BOS at position 0
        if hidden.shape[1] <= 1:
            embedding = hidden[0].mean(dim=0)
        else:
            embedding = hidden[0, 1:].mean(dim=0)
        return embedding.cpu().float().numpy()


def cross_validated_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(mean, std, max)`` accuracy from stratified k-fold logreg."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: list[float] = []
    for train_idx, test_idx in cv.split(X_scaled, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, n_jobs=-1)
        clf.fit(X_scaled[train_idx], y[train_idx])
        accs.append(float(clf.score(X_scaled[test_idx], y[test_idx])))
    arr = np.asarray(accs)
    return float(arr.mean()), float(arr.std()), float(arr.max())


def probe_layers(
    extractor: HiddenStateExtractor,
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    layers: Iterable[int] | None = None,
) -> list[LayerProbeResult]:
    """For each layer, extract embeddings and run cross-validated logreg."""
    layer_indices = list(layers) if layers is not None else list(range(extractor.num_layers))
    results: list[LayerProbeResult] = []
    for layer_idx in layer_indices:
        X = np.stack([extractor.embed_span(t, layer_idx=layer_idx) for t in texts])
        mean, std, mx = cross_validated_logreg(X, labels, n_splits=n_splits, seed=seed)
        results.append(LayerProbeResult(layer=layer_idx, mean_acc=mean, std_acc=std, max_acc=mx))
    return results


def probe_by_position(
    embeddings: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    *,
    n_buckets: int = 5,
    min_per_bucket: int = 30,
    n_splits: int = 5,
    seed: int = 42,
) -> list[PositionProbeResult]:
    """Slice spans into ``n_buckets`` position bins and probe each."""
    if not (len(embeddings) == len(labels) == len(positions)):
        raise ValueError("embeddings/labels/positions must have equal length")
    results: list[PositionProbeResult] = []
    for bucket in range(n_buckets):
        low = bucket / n_buckets
        high = (bucket + 1) / n_buckets
        mask = (positions >= low) & (positions < high)
        if mask.sum() < min_per_bucket:
            continue
        y = labels[mask]
        if (y == 1).mean() < 0.2 or (y == 0).mean() < 0.2:
            continue
        mean, std, _ = cross_validated_logreg(
            embeddings[mask], y, n_splits=n_splits, seed=seed
        )
        results.append(
            PositionProbeResult(
                bucket=bucket,
                position_label=f"{int(low * 100)}-{int(high * 100)}%",
                mean_acc=mean,
                std_acc=std,
                n_spans=int(mask.sum()),
            )
        )
    return results
