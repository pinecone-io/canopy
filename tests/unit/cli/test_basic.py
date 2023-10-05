import sys
from resin import __version__


def test_version():
    if sys.version_info > (3, 11):
        import tomllib as toml

        with open("pyproject.toml", "rb") as f:
            assert toml.load(f)["tool"]["poetry"]["version"] == __version__
    else:
        import toml

        with open("pyproject.toml") as f:
            assert toml.load(f)["tool"]["poetry"]["version"] == __version__
