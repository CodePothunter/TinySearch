#!/usr/bin/env python3
"""
Test script for the new TinySearch logging system
"""
import sys
from pathlib import Path

# Add tinysearch to path
sys.path.insert(0, str(Path(__file__).parent))

from tinysearch.logger import (
    get_logger, 
    configure_logger, 
    log_step, 
    log_progress, 
    log_success, 
    log_warning, 
    log_error
)

def test_modern_format():
    """Test modern format logging"""
    print("\n=== Testing Modern Format ===")
    
    configure_logger({
        "logging": {
            "level": "INFO",
            "format": "modern",
            "colorize": True,
            "show_time": True
        }
    })
    
    logger = get_logger("test_modern")
    
    logger.info("🚀 Starting TinySearch demo")
    log_step("Initializing components")
    logger.info("📄 Loading documents from data directory")
    log_progress("Processing documents", 3, 10)
    log_progress("Processing documents", 7, 10)
    log_progress("Processing documents", 10, 10)
    logger.info("✂️  Created 150 text chunks")
    logger.info("🧠 Generating embeddings with HuggingFace model")
    log_success("Index built successfully!")
    log_warning("GPU not available, using CPU")
    logger.info("🔍 Ready for queries")

def test_simple_format():
    """Test simple format logging"""
    print("\n=== Testing Simple Format ===")
    
    configure_logger({
        "logging": {
            "level": "INFO",
            "format": "simple",
            "colorize": True,
            "show_time": True
        }
    })
    
    logger = get_logger("test_simple")
    
    logger.info("Starting TinySearch demo")
    logger.info("Loading documents from data directory")
    logger.info("Created 150 text chunks")
    logger.info("Generating embeddings")
    log_success("Index built successfully!")
    log_warning("GPU not available, using CPU")

def test_detailed_format():
    """Test detailed format logging"""
    print("\n=== Testing Detailed Format ===")
    
    configure_logger({
        "logging": {
            "level": "DEBUG",
            "format": "detailed",
            "colorize": True,
            "show_time": True,
            "show_location": True
        }
    })
    
    logger = get_logger("test_detailed")
    
    logger.debug("Debug message with file location")
    logger.info("Info message with file location")
    logger.warning("Warning message with file location")
    logger.error("Error message with file location")

def test_error_handling():
    """Test error logging"""
    print("\n=== Testing Error Handling ===")
    
    configure_logger({
        "logging": {
            "level": "INFO",
            "format": "modern",
            "colorize": True
        }
    })
    
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except Exception as e:
        log_error("Failed to process document", e)

def main():
    """Run all logging tests"""
    print("🧪 Testing TinySearch Modern Logging System")
    print("=" * 50)
    
    test_modern_format()
    test_simple_format()
    test_detailed_format()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("✅ All logging tests completed!")

if __name__ == "__main__":
    main()
