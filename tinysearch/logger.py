"""
TinySearch Logger Configuration

This module provides a unified logging configuration using loguru with modern,
colorful output formats and flexible configuration options.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class TinySearchLogger:
    """
    Centralized logger configuration for TinySearch
    
    Provides modern, colorful logging with configurable levels and formats.
    """
    
    def __init__(self):
        self._configured = False
        self._default_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        self._simple_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<level>{message}</level>"
        )
    
    def configure(
        self,
        level: str = "INFO",
        format_style: str = "modern",
        show_time: bool = True,
        show_location: bool = False,
        file_output: Optional[str] = None,
        file_level: str = "DEBUG",
        colorize: bool = True
    ) -> None:
        """
        Configure the logger with specified options
        
        Args:
            level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_style: Format style ('modern', 'simple', 'detailed')
            show_time: Whether to show timestamps
            show_location: Whether to show file location info
            file_output: Optional file path for file logging
            file_level: Log level for file output
            colorize: Whether to use colored output
        """
        if self._configured:
            return
        
        # Remove default handler
        logger.remove()
        
        # Choose format based on style and options
        if format_style == "simple":
            console_format = self._simple_format
        elif format_style == "detailed" or show_location:
            console_format = self._default_format
        else:  # modern
            console_format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <5}</level> | "
                "<level>{message}</level>"
            )
        
        # Modify format based on options
        if not show_time:
            console_format = console_format.split(" | ", 1)[1]
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=colorize,
            backtrace=True,
            diagnose=True
        )
        
        # Add file handler if specified
        if file_output:
            file_path = Path(file_output)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=file_level,
                rotation="10 MB",
                retention="7 days",
                compression="gz",
                backtrace=True,
                diagnose=True
            )
        
        self._configured = True
    
    def get_logger(self, name: Optional[str] = None):
        """
        Get a logger instance
        
        Args:
            name: Optional logger name
            
        Returns:
            Logger instance
        """
        if not self._configured:
            self.configure()
        
        if name:
            return logger.bind(name=name)
        return logger


# Global logger instance
_logger_instance = TinySearchLogger()


def configure_logger(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure the global logger with settings from config
    
    Args:
        config: Configuration dictionary with logging settings
    """
    if config is None:
        config = {}
    
    # Extract logging configuration
    log_config = config.get("logging", {})
    
    _logger_instance.configure(
        level=log_config.get("level", "INFO"),
        format_style=log_config.get("format", "modern"),
        show_time=log_config.get("show_time", True),
        show_location=log_config.get("show_location", False),
        file_output=log_config.get("file", None),
        file_level=log_config.get("file_level", "DEBUG"),
        colorize=log_config.get("colorize", True)
    )


def get_logger(name: Optional[str] = None):
    """
    Get a configured logger instance
    
    Args:
        name: Optional logger name for identification
        
    Returns:
        Configured logger instance
    """
    return _logger_instance.get_logger(name)


# Convenience functions for common logging patterns
def log_progress(message: str, current: int, total: int) -> None:
    """Log progress with a modern format"""
    percentage = (current / total) * 100 if total > 0 else 0
    get_default_logger().info(f"{message} [{current}/{total}] ({percentage:.1f}%)")


def log_step(step_name: str, details: Optional[str] = None) -> None:
    """Log a processing step with consistent formatting"""
    if details:
        get_default_logger().info(f"🔄 {step_name}: {details}")
    else:
        get_default_logger().info(f"🔄 {step_name}")


def log_success(message: str) -> None:
    """Log a success message with emoji"""
    get_default_logger().success(f"✅ {message}")


def log_warning(message: str) -> None:
    """Log a warning message with emoji"""
    get_default_logger().warning(f"⚠️  {message}")


def log_error(message: str, exception: Optional[Exception] = None) -> None:
    """Log an error message with emoji and optional exception"""
    if exception:
        get_default_logger().error(f"❌ {message}: {exception}")
    else:
        get_default_logger().error(f"❌ {message}")


# Export the main logger for direct use
# Note: This creates a default logger instance
_default_logger = None

def get_default_logger():
    """Get the default logger instance"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    return _default_logger
