"""Make the flat top-level modules importable from tests/ regardless of cwd."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
