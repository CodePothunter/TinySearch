#!/usr/bin/env python
"""
TinySearch Web UI Demo

This script sets up a simple demo with sample data and launches the web UI.
"""
import os
import sys
import yaml
from pathlib import Path
import shutil
import tempfile
import argparse
import subprocess
import time
import webbrowser

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from tinysearch.config import Config
from tinysearch.cli import load_adapter, load_splitter, load_embedder, load_indexer, load_query_engine
from tinysearch.flow.controller import FlowController


def create_sample_data(sample_dir):
    """Create sample documents for the demo"""
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample 1: Python tutorial
    with open(os.path.join(sample_dir, "python_tutorial.txt"), "w") as f:
        f.write("""# Python Tutorial

## Introduction
Python is a high-level, interpreted programming language that is easy to learn and use.
It has a clean and readable syntax, making it a great language for beginners.
Python is widely used in data science, web development, automation, and machine learning.

## Basic Syntax
Python uses indentation to define code blocks. Here's a simple example:

```python
def greet(name):
    print(f"Hello, {name}!")
    
greet("World")  # Output: Hello, World!
```

## Data Types
Python has several built-in data types:
- Numbers (int, float, complex)
- Strings
- Lists
- Tuples
- Dictionaries
- Sets

## Control Flow
Python supports standard control flow statements:
- if/elif/else conditions
- for and while loops
- try/except for error handling
""")
    
    # Sample 2: Machine Learning overview
    with open(os.path.join(sample_dir, "machine_learning.txt"), "w") as f:
        f.write("""# Machine Learning Overview

## Introduction
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
Instead of explicitly programming rules, machine learning algorithms identify patterns in data and make predictions.

## Types of Machine Learning
1. Supervised Learning - The algorithm learns from labeled training data
2. Unsupervised Learning - The algorithm finds patterns in unlabeled data
3. Reinforcement Learning - The algorithm learns through rewards and punishments

## Common Algorithms
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks
- K-means Clustering

## Applications
Machine learning is used in many fields including:
- Image and speech recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Self-driving cars
- Medical diagnosis
""")
    
    # Sample 3: Vector Databases
    with open(os.path.join(sample_dir, "vector_databases.txt"), "w") as f:
        f.write("""# Vector Databases

## Introduction
Vector databases are specialized database systems designed to store and query vector embeddings efficiently.
They are crucial for applications involving semantic search, recommendation systems, and machine learning.

## Key Features
- Vector indexing for similarity search
- Efficient nearest neighbor algorithms
- Support for high-dimensional vector spaces
- Specialized distance metrics (cosine, Euclidean, etc.)

## Popular Vector Databases
1. FAISS - Facebook AI Similarity Search
2. Pinecone
3. Milvus
4. Qdrant
5. Weaviate
6. Chroma

## Vector Search Methods
- Exact search methods like brute force
- Approximate methods like HNSW (Hierarchical Navigable Small World)
- Quantization-based methods
- Tree-based methods like KD-trees

## Integration
Vector databases are commonly used in:
- RAG (Retrieval-Augmented Generation) systems
- Semantic search engines
- Content recommendation platforms
- Image similarity search
- Anomaly detection systems
""")
    
    return sample_dir


def create_demo_config():
    """Create a configuration file for the demo"""
    config = {
        "adapter": {
            "type": "text",
            "params": {
                "encoding": "utf-8"
            }
        },
        "splitter": {
            "type": "character",
            "chunk_size": 200,
            "chunk_overlap": 50,
            "separator": "\n\n"
        },
        "embedder": {
            "type": "huggingface",
            "model": "Qwen/Qwen-Embedding",
            "device": "cpu",
            "normalize": True
        },
        "indexer": {
            "type": "faiss",
            "index_path": ".cache/index.faiss",
            "metric": "cosine"
        },
        "query_engine": {
            "method": "template",
            "template": "{query}",
            "top_k": 5
        },
        "flow": {
            "use_cache": True,
            "cache_dir": ".cache"
        }
    }
    
    # Save the configuration to a temporary file
    config_path = os.path.join(tempfile.gettempdir(), "tinysearch_demo_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


def build_demo_index(config_path, data_dir):
    """Build the search index for the demo"""
    # Load configuration
    config = Config(config_path)
    config_dict = config.config
    
    # Load components
    data_adapter = load_adapter(config)
    text_splitter = load_splitter(config)
    embedder = load_embedder(config)
    indexer = load_indexer(config)
    query_engine = load_query_engine(config, embedder, indexer)
    
    # Initialize flow controller
    flow_controller = FlowController(
        data_adapter=data_adapter,
        text_splitter=text_splitter,
        embedder=embedder,
        indexer=indexer,
        query_engine=query_engine,
        config=config_dict
    )
    
    # Build index
    print(f"Building index from {data_dir}...")
    flow_controller.build_index(data_path=data_dir)
    
    # Save index
    print("Saving index...")
    flow_controller.save_index()
    
    print("Index built successfully!")


def start_api_server(config_path):
    """Start the API server with the demo configuration"""
    env = os.environ.copy()
    env["TINYSEARCH_CONFIG"] = config_path
    
    # Start the API server as a subprocess
    print("Starting TinySearch API server...")
    process = subprocess.Popen(
        [sys.executable, "-m", "tinysearch.api"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to start
    time.sleep(2)
    
    # Open the web browser
    print("Opening web browser to http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    
    print("\nPress Ctrl+C to stop the server")
    try:
        # Keep the server running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description="TinySearch Web UI Demo")
    parser.add_argument("--skip-index", action="store_true", help="Skip building the index (use existing)")
    args = parser.parse_args()
    
    # Create sample data
    print("Creating sample data...")
    sample_dir = create_sample_data(os.path.join(tempfile.gettempdir(), "tinysearch_demo_data"))
    
    # Create configuration
    config_path = create_demo_config()
    print(f"Configuration saved to {config_path}")
    
    # Build index if not skipped
    if not args.skip_index:
        build_demo_index(config_path, sample_dir)
    
    # Start the API server
    start_api_server(config_path)


if __name__ == "__main__":
    main() 