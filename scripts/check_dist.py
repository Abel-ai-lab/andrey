"""Release gate: every file in the built sdist and wheel is on the packaging allowlist.

Both artifacts carry the ``andrey`` package's Python source and the metadata files the build adds,
nothing else: no tests, QA tools, docs, benchmarks, demos, notebooks, or recovery baselines. Any
other file fails the check, so shipping a data file means widening the allowlist here on purpose.
The wheel and the sdist must also carry the same package modules, and the wheel's metadata version
must equal ``andrey.__version__``.

``--readme`` also requires every link and image in the package description to be an absolute URL:
the package index renders the README without the repository beside it.

    uv build --out-dir dist
    uv run --no-project python scripts/check_dist.py dist [--list] [--readme]
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

# Wheel: the import package plus the dist-info files hatchling writes.
_WHEEL_DIST_INFO = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"}
# Sdist, below its top directory: the package source plus the files hatchling always adds.
_SDIST_ROOT_FILES = {"pyproject.toml", "README.md", "LICENSE", "PKG-INFO", ".gitignore"}

_FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)
_MD_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_REF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
_HTML_ATTR = re.compile(r"\b(href|src|srcset)\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)
_ABSOLUTE = ("https://", "http://", "mailto:")
_VERSION = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


def _package_module(path: str) -> str | None:
    """Return ``path`` if it is a Python source file of the ``andrey`` package."""
    parts = PurePosixPath(path).parts
    return path if parts[:1] == ("andrey",) and path.endswith(".py") else None


def wheel_files(wheel: Path) -> list[str]:
    """Every file name in ``wheel``, sorted."""
    with zipfile.ZipFile(wheel) as archive:
        return sorted(n for n in archive.namelist() if not n.endswith("/"))


def sdist_files(sdist: Path) -> list[str]:
    """Every file in ``sdist`` relative to its top directory, sorted."""
    with tarfile.open(sdist) as archive:
        names = [m.name for m in archive.getmembers() if not m.isdir()]
    return sorted(PurePosixPath(*PurePosixPath(n).parts[1:]).as_posix() for n in names)


def check_wheel(names: list[str]) -> list[str]:
    """Wheel files outside the allowlist."""
    bad = []
    for name in names:
        head, _, rest = name.partition("/")
        if _package_module(name) or (head.endswith(".dist-info") and rest in _WHEEL_DIST_INFO):
            continue
        bad.append(name)
    return bad


def check_sdist(names: list[str]) -> list[str]:
    """Sdist files outside the allowlist."""
    bad = []
    for name in names:
        if name in _SDIST_ROOT_FILES:
            continue
        if name.startswith("src/") and _package_module(name.removeprefix("src/")):
            continue
        bad.append(name)
    return bad


def module_mismatch(wheel_names: list[str], sdist_names: list[str]) -> list[str]:
    """Package modules present in only one artifact."""
    in_wheel = {n for n in wheel_names if _package_module(n)}
    in_sdist = {n.removeprefix("src/") for n in sdist_names if n.startswith("src/")}
    in_sdist = {n for n in in_sdist if _package_module(n)}
    return sorted(in_wheel ^ in_sdist)


def version_mismatch(wheel: Path) -> str | None:
    """Describe a difference between the wheel's metadata version and ``andrey.__version__``."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        declared = Parser().parsestr(archive.read(name).decode("utf-8")).get("Version")
        match = _VERSION.search(archive.read("andrey/__init__.py").decode("utf-8"))
    found = match[1] if match else None
    if declared == found:
        return None
    return f"metadata version {declared} but andrey.__version__ {found}"


def relative_links(markdown: str) -> list[str]:
    """Link and image targets in ``markdown`` that are not absolute URLs, outside code fences."""
    text = _FENCE.sub("", markdown)
    targets = [*_MD_LINK.findall(text), *_MD_REF.findall(text)]
    for attr, value in _HTML_ATTR.findall(text):
        if attr.lower() == "srcset":
            targets += [c.split()[0] for c in value.split(",") if c.strip()]
        else:
            targets.append(value)
    return [t for t in targets if not t.startswith(_ABSOLUTE)]


def wheel_description(wheel: Path) -> str:
    """Read the long description from the wheel's METADATA."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        message = Parser().parsestr(archive.read(name).decode("utf-8"))
    return str(message.get_payload())


def _one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {directory}, found {len(found)}")
    return found[0]


def main(argv: list[str] | None = None) -> int:
    """Check the artifacts in ``dist``; return 1 on any packaging problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, help="directory holding one wheel and one sdist")
    parser.add_argument("--list", action="store_true", help="print every file in both artifacts")
    parser.add_argument("--readme", action="store_true", help="require absolute README links")
    args = parser.parse_args(argv)

    wheel, sdist = _one(args.dist, "*.whl"), _one(args.dist, "*.tar.gz")
    in_wheel, in_sdist = wheel_files(wheel), sdist_files(sdist)
    if args.list:
        for artifact, names in ((wheel, in_wheel), (sdist, in_sdist)):
            print(f"{artifact.name}: {len(names)} files")
            print("\n".join(f"  {n}" for n in names))

    failures = [f"wheel: {n}" for n in check_wheel(in_wheel)]
    failures += [f"sdist: {n}" for n in check_sdist(in_sdist)]
    failures += [f"in one artifact only: {n}" for n in module_mismatch(in_wheel, in_sdist)]
    if mismatch := version_mismatch(wheel):
        failures.append(mismatch)
    if args.readme:
        failures += [
            f"README link is not absolute: {t}" for t in relative_links(wheel_description(wheel))
        ]
    if failures:
        print(f"{len(failures)} packaging problem(s):", file=sys.stderr)
        print("\n".join(f"  {f}" for f in failures), file=sys.stderr)
        return 1
    print(f"clean: {wheel.name} ({len(in_wheel)} files), {sdist.name} ({len(in_sdist)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
