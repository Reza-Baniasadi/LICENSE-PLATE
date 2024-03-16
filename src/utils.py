from pathlib import Path


def mkdir_incremental(path: Path):
    path = Path(path)
    i = 0
    new_path = path
    while new_path.exists():
        i += 1
        new_path = path.parent / f"{path.name}_{i}"
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path
