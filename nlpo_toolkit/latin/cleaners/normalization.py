import re


def normalize_cleaned_text(lines: tuple[str, ...]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip() + "\n"
