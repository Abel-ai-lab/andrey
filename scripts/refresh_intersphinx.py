"""Download again the stored intersphinx inventories that the docs build falls back on.

``docs/conf.py`` reads each inventory in ``intersphinx_mapping`` from its project's site first and
from its copy in ``docs/_intersphinx/`` second. This script downloads every inventory from its site,
checks that each download parses as an inventory, and only then replaces the copies, so a failed
download changes none of them. It reads ``docs/conf.py``, so it runs with the docs group:

    uv run --group docs python scripts/refresh_intersphinx.py
"""

from __future__ import annotations

import posixpath
import runpy
import sys
import urllib.request
from pathlib import Path

from sphinx.util.inventory import InventoryFile

DOCS = Path(__file__).resolve().parents[1] / "docs"
TIMEOUT = 60  # seconds, for each connection attempt and each read


def mapping() -> dict[str, tuple[str, tuple[str | None, ...]]]:
    """Return ``intersphinx_mapping`` from ``docs/conf.py``."""
    path = list(sys.path)  # conf.py puts apps/ on the path
    try:
        return runpy.run_path(str(DOCS / "conf.py"))["intersphinx_mapping"]
    finally:
        sys.path[:] = path


def download(url: str) -> bytes:
    """Return the body of ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": "andrey-docs"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def main() -> int:
    """Download every inventory, then replace the stored copies."""
    downloads = {}
    for name, (site, locations) in mapping().items():
        url = posixpath.join(site, "objects.inv")
        raw = download(url)
        InventoryFile.loads(raw, uri=site)  # raises on a page that is not an inventory
        downloads[DOCS / locations[-1]] = raw
        print(f"{name}: {len(raw)} bytes from {url}")
    for path, raw in downloads.items():
        path.write_bytes(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
