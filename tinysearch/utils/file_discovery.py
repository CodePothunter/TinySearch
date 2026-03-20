"""
Centralized file discovery for directory-based indexing.
"""
import logging
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# Default file extensions per adapter type
ADAPTER_EXTENSIONS = {
    "text": [".txt", ".text", ".md", ".py", ".js", ".html", ".css", ".json"],
    "pdf": [".pdf"],
    "csv": [".csv"],
    "markdown": [".md", ".markdown", ".mdown", ".mkdn"],
    "json": [".json"],
}


def iter_input_files(
    data_path: Path,
    adapter_type: str = "text",
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> Iterator[Path]:
    """
    Discover files under a path that a given adapter can process.

    Args:
        data_path: File or directory path
        adapter_type: Adapter type name, used to look up default extensions
        extensions: Custom extension list (overrides defaults)
        recursive: Whether to recurse into subdirectories

    Yields:
        Path objects in sorted order for deterministic results
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Path not found: {data_path}")

    if data_path.is_file():
        yield data_path
        return

    # Directory: filter by extensions
    allowed = set(
        ext.lower() for ext in (extensions or ADAPTER_EXTENSIONS.get(adapter_type, []))
    )

    pattern = data_path.rglob("*") if recursive else data_path.iterdir()
    for child in sorted(pattern):
        if child.is_file() and (not allowed or child.suffix.lower() in allowed):
            yield child
