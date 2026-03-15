"""
Load API keys from secrets.env (or .env) into os.environ.
Use this at the start of scripts/notebooks so keys are available without hardcoding.

Usage:
    from load_secrets import load_secrets
    load_secrets()   # loads from secrets.env or .env in current dir / class3_runs

Or from shell before running:
    set -a && source secrets.env && set +a
    # or: export $(grep -v '^#' secrets.env | xargs)
"""
from pathlib import Path


def load_secrets(
    filename: str = "secrets.env",
    search_dirs: tuple[Path, ...] | None = None,
    optional: bool = True,
) -> bool:
    """
    Load KEY=VALUE lines from filename into os.environ.
    Skips empty lines and lines starting with #.
    search_dirs: directories to look for filename (default: cwd and class3_runs).
    optional: if True, do not raise when file is missing; return False.
    Returns True if file was found and loaded, False otherwise.
    """
    import os

    if search_dirs is None:
        cwd = Path.cwd()
        class3 = Path(__file__).resolve().parent
        search_dirs = (cwd, class3)

    for d in search_dirs:
        path = d / filename
        if path.is_file():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip()
                        if key and value and not key.startswith("#"):
                            os.environ[key] = value
            return True

    if not optional:
        raise FileNotFoundError(f"{filename} not found in {search_dirs}")
    return False
