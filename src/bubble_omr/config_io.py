# src/bubble_omr/config_io.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

def load_config_any(path: str | Path) -> Dict[str, Any]:
    """
    Prefer YAML, but transparently accept JSON.
    - If extension is .yml/.yaml -> use YAML
    - If extension is .json -> use JSON
    - Otherwise: try YAML first, then JSON
    """
    p = Path(path)
    data = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()

    if ext in {".yml", ".yaml"}:
        try:
            import yaml  # PyYAML
        except Exception as e:
            raise RuntimeError("YAML config provided but PyYAML is not installed. Install with: pip install pyyaml") from e
        cfg = yaml.safe_load(data)
    elif ext == ".json":
        cfg = json.loads(data)
    else:
        # No/unknown extension: prefer YAML, then fallback to JSON
        try:
            import yaml
            cfg = yaml.safe_load(data)
        except Exception:
            cfg = json.loads(data)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/object.")

    return cfg