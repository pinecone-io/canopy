from pathlib import Path


class Directory:
    """Stores the directory paths for Canopy library"""

    ROOT = Path(__file__).parent.parent
    CONFIG_TEMPLATES = ROOT.joinpath("config_templates")
