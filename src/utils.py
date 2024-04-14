from pathlib import Path

def create_unique_directory(base_path: Path):
    base_path = Path(base_path)
    counter = 0
    unique_path = base_path

    while unique_path.exists():
        counter += 1
        unique_path = base_path.parent / f"{base_path.stem}_{counter}"

    unique_path.mkdir(parents=True, exist_ok=True)
    return unique_path
