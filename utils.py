from pathlib import Path

def load_lemma_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}

    mapping = {}
    p = Path(path)
    if not p.exists():
        return {}

    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, dst = line.split("\t")
            mapping[src.strip()] = dst.strip()

    return mapping