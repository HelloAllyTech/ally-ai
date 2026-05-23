import os
from functools import lru_cache
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Singleton manager for loading prompt templates from the filesystem.
    Prompts are stored in app/prompts/<file_key>/<prompt_key>.txt
    """

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent / "prompts"
        logger.info(f"Initializing PromptManager with base_dir: {self.base_dir}")

    @lru_cache(maxsize=128)
    def get_template(self, file_key: str, prompt_key: str) -> str:
        """
        Retrieves a prompt template from the filesystem.
        Uses LRU cache to avoid frequent disk I/O.
        Note: Because of lru_cache, templates will NOT hot-reload if changed
        on disk while the server is running. You must call clear_cache()
        or restart the server to see changes.
        """
        file_path = self.base_dir / file_key / f"{prompt_key}.txt"

        if not file_path.exists():
            logger.error(f"Prompt template not found at {file_path}")
            return ""

        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error(f"Error reading prompt template at {file_path}: {e}")
            return ""

    def clear_cache(self):
        """Clears the template cache."""
        self.get_template.cache_clear()


# Singleton instance
prompt_manager = PromptManager()
