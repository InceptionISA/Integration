from pathlib import Path


class ConfigManager:
    """Manages configuration and paths for the submission process."""

    def __init__(self,
                 base_dir: str = ".",
                 competition_name: str = "surveillance-for-retail-stores") -> None:
        """
        Initialize the ConfigManager.

        Args:
            base_dir: Base directory for all operations
            competition_name: Name of the Kaggle competition
        """
        self.base_dir = Path(base_dir)
        self.competition_name = competition_name

        # Define paths
        self.paths = {
            'track': self.base_dir / 'submissions' / 'Track',
            'face': self.base_dir / 'submissions' / 'Face',
            'main': self.base_dir / 'submissions',
            'experiments': self.base_dir / 'experiments',
            'logs': self.base_dir / 'logs'
        }

        # Create directories if they don't exist
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        """Get a path from the configuration."""
        if key not in self.paths:
            raise ValueError(f"Unknown path key: {key}")
        return self.paths[key]
