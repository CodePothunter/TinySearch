#!/usr/bin/env python3
"""
Simple test script for the new TinySearch logging system (no external dependencies)
"""
import sys
from pathlib import Path

# Add tinysearch to path
sys.path.insert(0, str(Path(__file__).parent))

def test_logger_import():
    """Test that we can import the logger module"""
    try:
        from tinysearch.logger import (
            get_logger, 
            configure_logger, 
            log_step, 
            log_progress, 
            log_success, 
            log_warning, 
            log_error
        )
        print("✅ Successfully imported logger module")
        return True
    except Exception as e:
        print(f"❌ Failed to import logger module: {e}")
        return False

def test_basic_logging():
    """Test basic logging functionality"""
    try:
        from tinysearch.logger import get_logger, configure_logger, log_success
        
        # Configure logger
        configure_logger({
            "logging": {
                "level": "INFO",
                "format": "modern",
                "colorize": True
            }
        })
        
        # Get logger and test
        logger = get_logger("test")
        logger.info("🧪 Testing basic logging functionality")
        log_success("Basic logging test passed!")
        
        print("✅ Basic logging test passed")
        return True
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        return False

def test_config_integration():
    """Test config integration"""
    try:
        from tinysearch.config import Config
        
        # Test that config includes logging section
        config = Config()
        logging_config = config.get("logging", {})
        
        if logging_config:
            print("✅ Config includes logging section")
            print(f"   Default level: {logging_config.get('level')}")
            print(f"   Default format: {logging_config.get('format')}")
            return True
        else:
            print("❌ Config missing logging section")
            return False
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing TinySearch Logging System Integration")
    print("=" * 50)
    
    tests = [
        ("Logger Import", test_logger_import),
        ("Basic Logging", test_basic_logging),
        ("Config Integration", test_config_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TinySearch logging system is ready!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
