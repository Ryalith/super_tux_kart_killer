from pathlib import Path

__DATA_FOLDER = Path.home() / ".cache" / "stk_killer"


def set_data_folder(path: Path):
    global __DATA_FOLDER
    __DATA_FOLDER = path


def get_data_folder() -> Path:
    return __DATA_FOLDER
