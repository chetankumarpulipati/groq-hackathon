"""
Error handling and custom exceptions for the healthcare system.
"""

from typing import Optional, Dict, Any
from utils.logging import get_logger

logger = get_logger("error_handler")


class HealthcareSystemError(Exception):
    """Base exception for all healthcare system errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
        logger.error(f"HealthcareSystemError: {message} (Code: {error_code})")


class AudioProcessingError(HealthcareSystemError):
    """Exception raised for audio processing errors."""
    pass


class VisionProcessingError(HealthcareSystemError):
    """Exception raised for vision processing errors."""
    pass


class DataValidationError(HealthcareSystemError):
    """Exception raised for data validation errors."""
    pass


class DiagnosticError(HealthcareSystemError):
    """Exception raised for diagnostic processing errors."""
    pass


class NotificationError(HealthcareSystemError):
    """Exception raised for notification service errors."""
    pass


class MCPConnectionError(HealthcareSystemError):
    """Exception raised for MCP connection errors."""
    pass


class GroqAPIError(HealthcareSystemError):
    """Exception raised for Groq API errors."""
    pass


def handle_exception(func):
    """Decorator for handling exceptions with logging and retry logic."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HealthcareSystemError:
            # Re-raise healthcare system errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            raise HealthcareSystemError(
                f"Unexpected error in {func.__name__}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )

    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper
    return decorator
