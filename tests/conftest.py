"""Use a fresh project-local pytest temp root when Windows' shared temp is locked."""

from pathlib import Path
import tempfile


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[1] / ".test-runs"
        root.mkdir(exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix="pytest-", dir=root)
