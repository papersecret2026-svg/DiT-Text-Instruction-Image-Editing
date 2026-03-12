import os
from . import model_control
from . import regional_mask
from . import attention_utils
from . import controller

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_file(txt_path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]

    for enc in encodings:
        try:
            with open(txt_path, "r", encoding=enc) as f:
                contents = f.readlines()
                return contents
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Failed to read file with known encodings: {txt_path}")
