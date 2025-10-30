from enum import Enum


class PHIEvents(str, Enum):
    """PHI event types."""

    DATA_ACCESSED = "DATA_ACCESSED"
    DATA_MODIFIED = "DATA_MODIFIED"
    DATA_DELETED = "DATA_DELETED"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_EVENT = "SYSTEM_EVENT"  # Added for system events like processor start/stop


# Event descriptions mapping
PHI_EVENT_DESCRIPTIONS = {
    PHIEvents.DATA_ACCESSED: "Data accessed",
    PHIEvents.DATA_MODIFIED: "Data modified",
    PHIEvents.DATA_DELETED: "Data deleted",
    PHIEvents.SYSTEM_ERROR: "System error occurred",
    PHIEvents.SYSTEM_EVENT: "System event occurred",
}
