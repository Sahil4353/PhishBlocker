import re

def normalize_body(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s
