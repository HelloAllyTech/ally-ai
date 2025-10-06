import contextvars
from typing import Optional, Dict, Any

# Context variable for request metadata
request_metadata_var = contextvars.ContextVar("request_metadata", default=None)


class ExecutionManager:
    """Simplified execution context manager for request metadata only."""
    
    @staticmethod
    def get_request_metadata() -> Optional[Dict[str, Any]]:
        """Get the request metadata."""
        return request_metadata_var.get()
    
    @staticmethod
    def set_request_metadata(metadata: Dict[str, Any]) -> None:
        """Set the request metadata."""
        request_metadata_var.set(metadata)
