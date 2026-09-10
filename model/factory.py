from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .ms_tcn2 import MS_TCN2


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "MS_TCN2": MS_TCN2,
}

MS_TCN2_PARAM_ALIASES = {
    "num_layers_pg": "num_layers_PG",
    "num_layers_r": "num_layers_R",
    "num_refinement_stages": "num_R",
    "num_feature_maps": "num_f_maps",
    "input_dim": "dim",
}


def _normalize_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if name == "MS_TCN2":
        for alias, canonical in MS_TCN2_PARAM_ALIASES.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized.pop(alias)
            elif alias in normalized:
                normalized.pop(alias)
        normalized.pop("dropout", None)
    return normalized


def build_model(name: str, params: Mapping[str, Any]) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from exc
    return model_class(**_normalize_params(name, params))
