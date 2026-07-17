from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .constants import ANALYSIS_BEHAVIOR_VERSION, ANALYSIS_CACHE_SCHEMA_VERSION
from .codec import analysis_fingerprint_to_json_value
from .models import AnalysisFingerprint, CacheObjectPaths


def prepared_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_analysis_cache_key(
    *, prepared_text_sha256: str, fingerprint: AnalysisFingerprint
) -> str:
    payload = {
        "prepared_text_sha256": prepared_text_sha256,
        "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "behavior_version": ANALYSIS_BEHAVIOR_VERSION,
        "fingerprint": analysis_fingerprint_to_json_value(fingerprint),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_metadata_path(payload_path: Path) -> Path:
    return payload_path.with_name(f"{payload_path.stem}.meta.json")


def cache_object_paths(cache_dir: Path, cache_key: str) -> CacheObjectPaths:
    root = Path(cache_dir)
    payload = root / "objects" / cache_key[:2] / f"{cache_key}.jsonl"
    return CacheObjectPaths(
        payload=payload,
        metadata=cache_metadata_path(payload),
        lock=root / "locks" / cache_key[:2] / f"{cache_key}.lock",
    )
