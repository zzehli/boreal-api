class Utils:
    @staticmethod
    def get_url_from_path(path: str) -> str:
        base_url = "https://www.madewithnestle.ca"
        # Get just the filename from the path
        filename = path.split('/')[-1]
        # Remove .md extension if it exists
        filename = filename.replace('.md', '')
        # Replace underscores with forward slashes
        filename = filename.replace('_', '/')
        return f"{base_url}/{filename}"
    
    @staticmethod
    def get_all_brands() -> list[str]:
        """Get all brands as a flat array of strings."""
        from .constants import (
            AERO,
            HAAGEN_DAZS,
            KITKAT,
            NATURES_BOUNTY,
            NESCAFE,
            STARBUCKS,
        )
        return [brand for brand_list in [
        AERO, NESCAFE, NATURES_BOUNTY, KITKAT,
        HAAGEN_DAZS, STARBUCKS
        ] for brand in brand_list]