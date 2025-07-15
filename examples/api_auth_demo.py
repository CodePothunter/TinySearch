#!/usr/bin/env python3
"""
TinySearch API Authentication and Rate Limiting Demo

This script demonstrates how to:
1. Start the TinySearch API server with authentication and rate limiting enabled
2. Create and use API keys for authentication
3. Test rate limiting functionality
"""

import os
import time
import yaml
import json
import tempfile
import requests
from pathlib import Path

# Constants
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"
TEST_DATA_DIR = Path("example_data")
CONFIG_FILE = Path("api_config.yaml")

# Sample configuration with authentication and rate limiting
CONFIG_CONTENT = """
adapter:
  type: text
  params: {}
splitter:
  chunk_size: 300
  chunk_overlap: 50
embedder:
  model: Qwen/Qwen-Embedding
  device: cpu
  params: {}
indexer:
  index_path: index.faiss
  metric: cosine
query_engine:
  method: template
  template: "请帮我查找：{query}"
  top_k: 5
flow:
  use_cache: true
  cache_dir: .cache
api:
  auth_enabled: true
  default_key: "demo-api-key-12345"
  master_key: "demo-master-key-67890"
  rate_limit_enabled: true
  rate_limit: 5  # Very low for demonstration
  rate_limit_window: 10  # 10 seconds window
"""

def print_header(message):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def start_api_server():
    """
    Start the TinySearch API server as a background process.
    Returns the process object.
    """
    print_header("Starting TinySearch API server")
    
    # Save configuration to temporary file
    with open(CONFIG_FILE, "w") as f:
        f.write(CONFIG_CONTENT)
    
    print(f"Saved configuration to {CONFIG_FILE}")
    print(f"API authentication enabled with default key: demo-api-key-12345")
    print(f"API rate limiting set to 5 requests per 10 seconds")
    
    # Start the server (this is a placeholder - in a real script you would use subprocess)
    print(f"\nTo start the server, run in another terminal:")
    print(f"TINYSEARCH_CONFIG='{CONFIG_FILE}' python -m tinysearch.api --port {API_PORT}")
    
    # Note: In a real script, you would use:
    # import subprocess
    # process = subprocess.Popen(
    #     ["python", "-m", "tinysearch.api", "--port", str(API_PORT)],
    #     env={**os.environ, "TINYSEARCH_CONFIG": str(CONFIG_FILE)}
    # )
    # time.sleep(2)  # Give the server time to start
    # return process

def test_authentication():
    """Test API authentication with valid and invalid keys."""
    print_header("Testing API authentication")
    
    # Test with no API key
    print("1. Testing request with no API key...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"  Health check (no auth required): {response.status_code}, {response.json()}")
        
        response = requests.get(f"{API_URL}/index/stats")
        print(f"  Stats request (auth required): {response.status_code}")
        if response.status_code == 401:
            print("  ✅ Authentication working - request denied without API key")
        else:
            print("  ❌ Authentication not working correctly")
    except Exception as e:
        print(f"  Error: {e}")

    # Test with invalid API key
    print("\n2. Testing request with invalid API key...")
    try:
        headers = {"X-API-Key": "invalid-key"}
        response = requests.get(f"{API_URL}/index/stats", headers=headers)
        print(f"  Status code: {response.status_code}")
        if response.status_code == 401:
            print("  ✅ Authentication working - request denied with invalid API key")
        else:
            print("  ❌ Authentication not working correctly")
    except Exception as e:
        print(f"  Error: {e}")

    # Test with valid API key
    print("\n3. Testing request with valid API key...")
    try:
        headers = {"X-API-Key": "demo-api-key-12345"}
        response = requests.get(f"{API_URL}/index/stats", headers=headers)
        print(f"  Status code: {response.status_code}")
        if response.status_code == 200:
            print("  ✅ Authentication working - request accepted with valid API key")
        else:
            print(f"  ❌ Authentication failed: {response.text}")
    except Exception as e:
        print(f"  Error: {e}")

def test_generate_api_key():
    """Test generating new API keys."""
    print_header("Testing API key generation")
    
    # Generate a new API key with the master key
    try:
        headers = {"master-key": "demo-master-key-67890"}
        response = requests.post(f"{API_URL}/api-key?expires_in_days=1", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            api_key = data.get("api_key")
            expires_at = data.get("expires_at")
            print(f"✅ Generated new API key: {api_key}")
            print(f"   Expires at: {expires_at}")
            
            # Test the new key
            print("\nTesting the newly generated API key...")
            headers = {"X-API-Key": api_key}
            response = requests.get(f"{API_URL}/index/stats", headers=headers)
            if response.status_code == 200:
                print("✅ New API key works!")
            else:
                print(f"❌ New API key failed: {response.status_code}, {response.text}")
        else:
            print(f"❌ Failed to generate API key: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_rate_limiting():
    """Test rate limiting functionality."""
    print_header("Testing rate limiting")
    
    headers = {"X-API-Key": "demo-api-key-12345"}
    
    print(f"Making multiple requests to trigger rate limiting (limit: 5 requests per 10 seconds)...")
    for i in range(7):  # Intentionally exceed the rate limit
        try:
            start_time = time.time()
            response = requests.get(f"{API_URL}/index/stats", headers=headers)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"  Request {i+1}: ✅ Success ({elapsed:.2f}s)")
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                print(f"  Request {i+1}: ⛔ Rate limited ({elapsed:.2f}s, retry after: {retry_after}s)")
                print(f"  ✅ Rate limiting working correctly!")
            else:
                print(f"  Request {i+1}: ❌ Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"  Request {i+1}: ❌ Error: {e}")
        
        time.sleep(0.5)  # Small delay between requests
    
    print("\nWaiting for rate limit window to expire (10 seconds)...")
    time.sleep(10)
    
    print("Making another request after rate limit window...")
    try:
        response = requests.get(f"{API_URL}/index/stats", headers=headers)
        if response.status_code == 200:
            print(f"✅ Request successful after waiting - rate limiting works correctly!")
        else:
            print(f"❌ Request failed after waiting: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run the demo."""
    print_header("TinySearch API Authentication and Rate Limiting Demo")
    
    # Start API server (in a real script)
    # process = start_api_server()
    start_api_server()
    
    # Instructions for manual testing
    print("\nPlease start the server using the command above, then press Enter to continue...")
    input()
    
    try:
        # Run tests
        test_authentication()
        test_generate_api_key()
        test_rate_limiting()
    finally:
        # Clean up in a real script
        # if process:
        #     process.terminate()
        #     process.wait()
        
        # If you want to clean up the config file
        # os.remove(CONFIG_FILE)
        pass
    
    print_header("Demo completed!")

if __name__ == "__main__":
    main() 