from __future__ import annotations
from pathlib import Path
from .config_loader import load_clean_config
from . import clean_text
import sys


# Default config file. Modify this to switch the default config path.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Path = BASE_DIR.parent / "cleaners" / "config" / "sample.yml"


def _clean_single_file(input_path: Path, output_path: Path, kind: str) -> None:
    raw = input_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw, kind=kind)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    print(f"[{kind}] cleaned: {input_path} -> {output_path}")


def main(argv: list[str] | None = None) -> int:
    """
    Clean text(s) based on a YAML config file.

    Usage:
        python -m nlpo_toolkit.latin.cleaners.run_clean_config
        python -m nlpo_toolkit.latin.cleaners.run_clean_config path/to/config.yml

    YAML example (directory input + filename template):

        kind: corpus_corporum
        input: input
        output: output
        output_filename_template: "cleaned_{index:03d}.txt"

    Available template variables (for directory mode):
        {index} : Auto-incrementing index (1, 2, 3, ...)
        {stem}  : Original filename without extension
        {ext}   : Original file extension (without the dot)
    """
    if argv is None:
        argv = sys.argv[1:]

    # Determine which config file to use
    if argv:
        config_path = Path(argv[0])
    else:
        config_path = DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml_data = load_clean_config(config_path)
    print(f"[DEBUG] config_path={config_path}")

    kind = yaml_data["kind"]
    raw_input = yaml_data["input"]
    raw_output = yaml_data["output"]
    filename_template: str | None = yaml_data.get("output_filename_template")

    config_dir = config_path.parent

    # Resolve paths relative to the config file's directory
    input_path = Path(raw_input)
    if not input_path.is_absolute():
        input_path = (config_dir / input_path).resolve()

    output_path = Path(raw_output)
    if not output_path.is_absolute():
        output_path = (config_dir / output_path).resolve()

    # Directory mode
    if input_path.is_dir():
        # In directory mode, output must also be a directory
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                f"When input is a directory ({input_path}), "
                f"output must also be a directory, but got: {output_path}"
            )

        output_path.mkdir(parents=True, exist_ok=True)
        
        sources: list[Path] = []
        for p in sorted(input_path.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".txt":
                print(f"[SKIP] not .txt: {p}")
                continue
            sources.append(p)

        stems = [p.stem for p in sources]
        has_dup_stem = len(stems) != len(set(stems))

        effective_template: str | None
        if filename_template:
            if has_dup_stem:
                effective_template = filename_template
            else:
                effective_template = "{stem}.cleaned.{ext}"
        else:
            effective_template = None
        
        # Change to `rglob("*")` if recursive processing is needed
        idx = 0
        for src in sources:
            idx += 1

            if effective_template:
                stem = src.stem
                ext = src.suffix.lstrip(".")
                try:
                    name = effective_template.format(
                        index=idx,
                        stem=stem,
                        ext=ext,
                    )
                except Exception as e:
                    raise ValueError(
                    f"Invalid output_filename_template={effective_template!r} "
                    f"for file {src}: {e}"
                ) from e
            else:
                name = src.name
            
            dst = output_path / name
            _clean_single_file(src, dst, kind=kind)
        print(f"[{kind}] cleaned {idx} files in directory: {input_path} -> {output_path}")                  
    else:
        if output_path.is_dir():
            dst = output_path / input_path.name
        else:
            dst = output_path

        _clean_single_file(input_path, dst, kind=kind)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
