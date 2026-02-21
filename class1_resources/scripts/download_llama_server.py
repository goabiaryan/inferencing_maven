#!/usr/bin/env python3
"""
Download the official prebuilt llama.cpp server binary for this OS/arch.

Saves into class1_resources/llama_server_bin/ and prints LLAMA_SERVER_PATH.
Run from class1_resources: python scripts/download_llama_server.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

REPO = "ggml-org/llama.cpp"
# Where to extract (repo root / llama_server_bin)
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "llama_server_bin"


def _platform_key() -> str:
    """Return a short key used to match release assets (e.g. macos-arm64, win-x64)."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-x64"
    if system == "windows":
        return "win-x64" if "64" in machine or machine == "amd64" else "win-x86"
    return "ubuntu-x64"


def _asset_matches(asset_name: str, key: str) -> bool:
    name = asset_name.lower()
    if key == "macos-arm64":
        return "macos" in name and "arm64" in name and ("tar.gz" in name or "zip" in name)
    if key == "macos-x64":
        return "macos" in name and "x64" in name and ("tar.gz" in name or "zip" in name)
    if key.startswith("win-"):
        return "win" in name and ("x64" in name or "x86" in name) and "bin" in name
    if key == "ubuntu-x64":
        return ("ubuntu" in name or "linux" in name) and "x64" in name and "bin" in name
    return False


def _get_latest_release() -> tuple[str, list[dict]]:
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    tag = data["tag_name"]
    assets = data.get("assets", [])
    return tag, assets


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"Accept": "application/octet-stream"})
    with urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def main() -> int:
    key = _platform_key()
    print(f"Platform: {key}")
    print("Fetching latest release info...")
    try:
        tag, assets = _get_latest_release()
    except Exception as e:
        print(f"Failed to get release info: {e}", file=sys.stderr)
        return 1

    candidates = [a for a in assets if _asset_matches(a["name"], key)]
    if not candidates:
        # Fallback: any asset containing our key
        candidates = [a for a in assets if key.replace("-", "") in a["name"].lower()]
    if not candidates:
        print(f"No matching asset for {key}. Available:", [a["name"] for a in assets[:15]], file=sys.stderr)
        return 1

    asset = candidates[0]
    download_url = asset.get("browser_download_url") or (
        f"https://github.com/{REPO}/releases/download/{tag}/{asset['name']}"
    )
    archive_path = OUT_DIR / asset["name"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        print(f"Already have {asset['name']}; extracting...")
    else:
        print(f"Downloading {asset['name']}...")
        try:
            _download(download_url, archive_path)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 1

    # Extract (strip one level if archive is e.g. llama-tag-bin-macos-arm64/...)
    extract_to = OUT_DIR / "extracted"
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    else:
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(extract_to, filter="data")  # safe extraction (Python 3.12+)

    # Find llama-server or llama-server.exe (prefer bin/)
    server_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    all_matches = list(extract_to.rglob(server_name))
    server_path = next((p for p in all_matches if "bin" in str(p)), all_matches[0] if all_matches else None)
    if not server_path or not server_path.exists():
        print("Extracted but could not find llama-server binary.", file=sys.stderr)
        return 1

    # Copy the whole bin folder (binary + .dylib/.so deps) so the server can load libraries
    final_bin = OUT_DIR / "bin"
    if final_bin.exists():
        shutil.rmtree(final_bin)
    final_bin.mkdir(parents=True)
    for item in server_path.parent.iterdir():
        dest = final_bin / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            if item.name == server_name and platform.system() != "Windows":
                dest.chmod(0o755)
        else:
            shutil.copytree(item, dest)
    final_server = final_bin / server_name

    print(f"llama-server at: {final_server}")
    print()
    print("Set and verify (Step 2 in Serve_local_model.md):")
    print(f'  export LLAMA_SERVER_PATH="{final_server}"')
    print("  # then set LLAMA_MODEL_PATH (from Step 1) and run:")
    print("  python scripts/check_llama_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
