#!/usr/bin/env python3
"""
TinySearch FAISS GPU/CPU Demo

This script demonstrates how to configure TinySearch to use GPU acceleration
for FAISS indexing when available, and gracefully fall back to CPU when not.
"""

import numpy as np
from pathlib import Path
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.base import TextChunk

def check_gpu_support():
    """Check if FAISS GPU support is available"""
    try:
        import faiss
        if hasattr(faiss, 'get_num_gpus'):
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 0:
                print(f"✅ FAISS GPU support detected: {gpu_count} GPUs available")
                return True
            else:
                print(f"❌ FAISS reports no GPUs available (get_num_gpus = {gpu_count})")
        else:
            print("❌ FAISS GPU support not detected (get_num_gpus not found)")
        
        # Additional check for GPU methods
        if hasattr(faiss, 'StandardGpuResources'):
            print("✅ StandardGpuResources is available")
        else:
            print("❌ StandardGpuResources not found")
        
        return False
    except ImportError:
        print("❌ FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
        return False

def compare_indexers(dim=128, num_vectors=10000):
    """Compare CPU and GPU indexers performance"""
    # Create random vectors and text chunks for testing
    np_vectors = np.random.random((num_vectors, dim)).astype('float32')
    # Convert numpy arrays to Python lists for compatibility
    vectors = np_vectors.tolist()
    chunks = [TextChunk(f"Text chunk {i}", {"id": i}) for i in range(num_vectors)]
    
    # Create and test CPU indexer
    print("\n--- Testing CPU Indexer ---")
    cpu_indexer = FAISSIndexer(index_type="Flat", use_gpu=False)
    cpu_indexer.build(vectors, chunks)
    
    # Create and test GPU indexer
    print("\n--- Testing GPU Indexer ---")
    gpu_indexer = FAISSIndexer(index_type="Flat", use_gpu=True)
    gpu_indexer.build(vectors, chunks)
    
    # Perform a sample search with both indexers
    np_query = np.random.random(dim).astype('float32')
    query = np_query.tolist()  # Convert to list for API compatibility
    
    print("\n--- Search Performance Comparison ---")
    # CPU search
    import time
    start = time.time()
    cpu_results = cpu_indexer.search(query, top_k=5)
    cpu_time = time.time() - start
    print(f"CPU search time: {cpu_time:.6f} seconds")
    
    # GPU search (if available)
    start = time.time()
    gpu_results = gpu_indexer.search(query, top_k=5)
    gpu_time = time.time() - start
    print(f"GPU search time: {gpu_time:.6f} seconds")
    
    # Show speedup
    if gpu_indexer.use_gpu:
        print(f"Speedup: {cpu_time/gpu_time:.2f}x faster with GPU")

def main():
    """Main demo function"""
    print("=== TinySearch FAISS GPU/CPU Demo ===\n")
    
    # First, check if FAISS GPU support is available
    gpu_support = check_gpu_support()
    print(f"\nGPU Acceleration Available: {gpu_support}")
    
    # Show which package is installed
    try:
        import pkg_resources
        try:
            faiss_cpu = pkg_resources.get_distribution("faiss-cpu")
            print(f"Installed package: faiss-cpu {faiss_cpu.version}")
        except pkg_resources.DistributionNotFound:
            try:
                faiss_gpu = pkg_resources.get_distribution("faiss-gpu") 
                print(f"Installed package: faiss-gpu {faiss_gpu.version}")
            except pkg_resources.DistributionNotFound:
                print("No FAISS package detected")
    except ImportError:
        print("Could not check installed packages")
    
    # Show installation instructions
    if not gpu_support:
        print("\nTo enable GPU support:")
        print("1. Uninstall faiss-cpu: pip uninstall -y faiss-cpu")
        print("2. Install faiss-gpu: pip install faiss-gpu")
        print("Note: This requires CUDA to be installed on your system")
    
    # Compare indexers
    print("\nComparing CPU and GPU indexers...\n")
    compare_indexers()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 