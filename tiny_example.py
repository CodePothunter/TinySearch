#!/usr/bin/env python
"""
Minimal example of TinySearch functionality without using FlowController.
This demonstrates the core components directly.
"""
import os
import tempfile
from pathlib import Path

from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FaissIndexer
from tinysearch.query.template import TemplateQueryEngine

# Create sample documents
with tempfile.TemporaryDirectory() as temp_dir:
    # Create a few sample text files
    docs = {
        "python.txt": """
        Python is a high-level, general-purpose programming language. Its design philosophy 
        emphasizes code readability with the use of significant indentation. Python is dynamically 
        typed and garbage-collected. It supports multiple programming paradigms, including structured, 
        object-oriented, and functional programming.
        """,
        
        "machine_learning.txt": """
        Machine learning is a field of inquiry devoted to understanding and building methods that 
        'learn', that is, methods that leverage data to improve performance on some set of tasks. 
        It is seen as a part of artificial intelligence. Machine learning algorithms build a model 
        based on sample data, known as training data, in order to make predictions or decisions 
        without being explicitly programmed to do so.
        """,
        
        "vector_search.txt": """
        Vector search, also known as semantic search or similarity search, is a technique used to 
        find items in a dataset that are similar to a query item. Unlike traditional keyword-based 
        search, vector search converts items into high-dimensional vectors and measures their 
        similarity using distance metrics like cosine similarity or Euclidean distance.
        """
    }
    
    # Write the documents to files
    for filename, content in docs.items():
        file_path = Path(temp_dir) / filename
        with open(file_path, "w") as f:
            f.write(content)
    
    print(f"Created sample documents in {temp_dir}")
    
    # Initialize components
    adapter = TextAdapter()
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    
    # Use a lightweight embedding model for the example
    embedder = HuggingFaceEmbedder(
        model_name="Qwen/Qwen3-Embedding-0.6B",  # Smaller model for quick loading
        device="cpu"  # Use CPU for compatibility
    )
    
    indexer = FaissIndexer(metric="cosine")
    query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
    
    # Process the documents
    print("Processing documents...")
    for filename in docs.keys():
        file_path = Path(temp_dir) / filename
        
        # Extract text
        texts = adapter.extract(file_path)
        
        # Create metadata
        metadata = [{"source": filename} for _ in range(len(texts))]
        
        # Split text into chunks
        chunks = splitter.split(texts, metadata)
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed(chunk_texts)
        
        # Add to index
        indexer.build(vectors, chunks)
    
    print(f"Created index with {len(indexer.texts)} chunks")
    
    # Perform a search
    query = "How does vector search work?"
    print(f"\nSearching for: '{query}'")
    
    results = query_engine.retrieve(query, top_k=3)
    
    # Display results
    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
        print(f"Source: {result['chunk'].metadata['source']}")
        print(f"Text: {result['text']}")

print("\nExample completed successfully!") 