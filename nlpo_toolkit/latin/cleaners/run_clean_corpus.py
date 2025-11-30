from __future__ import annotations
from pathlib import Path
from .config_loader import load_clean_config
from . import clean_text
import sys


# Modify this to switch the default config file being used
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Path = BASE_DIR.parent / "cleaners" / "config" /"sample.yml"

def main(argv: list[str] | None = None) -> int:
    """
    Clean a text according to a YAML config.

    Usage:
        python -m nlpo_toolkit.latin.cleaners.run_clean_config
        python -m nlpo_toolkit.latin.cleaners.run_clean_config path/to/config.yml
    """
    if argv is None:
        argv = sys.argv[1:]

    # 1) Decide which config to use
    if argv:
        config_path = Path(argv[0])
    else:
        config_path = DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml_data = load_clean_config(config_path)

    kind = yaml_data["kind"]
    raw_input = yaml_data["input"]
    raw_output = yaml_data["output"]

    config_dir = config_path.parent

    # Resolve input/output relative to the config file directory
    input_path = Path(raw_input)
    if not input_path.is_absolute():
        input_path = (config_dir / input_path).resolve()

    output_path = Path(raw_output)
    if not output_path.is_absolute():
        output_path = (config_dir / output_path).resolve()

    raw = input_path.read_text(encoding="utf-8")

    cleaned = clean_text(raw, kind=kind)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"[{kind}] cleaned -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
