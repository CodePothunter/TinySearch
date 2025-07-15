#!/usr/bin/env python3
"""
A simple example of using TinySearch
"""
import os
from pathlib import Path
import argparse
import json
import sys

# Add parent directory to path to allow importing tinysearch from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinysearch.config import Config
from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.query.template import TemplateQueryEngine


def create_example_documents(directory: Path):
    """
    Create example documents for testing
    
    Args:
        directory: Directory to create documents in
    """
    os.makedirs(directory, exist_ok=True)
    
    # Create example documents
    documents = [
        {
            "filename": "ai_ethics.txt",
            "content": """
            Artificial Intelligence Ethics
            
            AI ethics is a set of principles that guide responsible development and deployment of AI systems.
            
            Key principles include:
            1. Fairness: AI systems should not discriminate against individuals or groups.
            2. Transparency: The decision-making process of AI should be explainable.
            3. Privacy: AI systems should respect user privacy and data protection.
            4. Safety: AI systems should operate reliably and safely.
            5. Accountability: There should be clear responsibility for AI systems' actions.
            
            Challenges in AI Ethics:
            - Bias in training data can lead to biased AI systems
            - Black-box models are difficult to explain
            - Privacy concerns when using personal data
            - Automation may lead to job displacement
            
            Organizations like IEEE, ACM, and various governmental bodies have proposed ethical guidelines for AI development.
            """
        },
        {
            "filename": "vector_databases.txt",
            "content": """
            Vector Databases
            
            Vector databases are specialized database systems designed to store and query vector embeddings efficiently.
            
            Key features of vector databases:
            1. Fast similarity search using approximate nearest neighbor algorithms
            2. Support for high-dimensional vectors
            3. Optimized for both accuracy and speed
            4. Scalability to handle large collections of vectors
            
            Popular vector database systems:
            - FAISS (Facebook AI Similarity Search)
            - Elasticsearch with vector search capability
            - Pinecone
            - Milvus
            - Weaviate
            
            Vector databases are essential components in AI applications involving semantic search, recommendation systems, and clustering.
            """
        },
        {
            "filename": "embedding_models.txt",
            "content": """
            Embedding Models
            
            Embedding models convert text, images, or other data into numerical vector representations.
            
            Types of embedding models:
            1. Word embeddings: Word2Vec, GloVe, FastText
            2. Sentence embeddings: SBERT, USE (Universal Sentence Encoder)
            3. Document embeddings: Doc2Vec
            4. Multilingual embeddings: mBERT, XLM-R
            
            Applications of embeddings:
            - Semantic search
            - Text classification
            - Clustering similar content
            - Information retrieval
            - Knowledge graphs
            
            Recent advancements include contextual embeddings from transformer models like BERT, GPT, and T5.
            """
        }
    ]
    
    # Write documents to files
    for doc in documents:
        with open(directory / doc["filename"], "w", encoding="utf-8") as f:
            f.write(doc["content"])
    
    return [doc["filename"] for doc in documents]


def main():
    """
    Main function
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="TinySearch simple example")
    parser.add_argument("--data", default="example_data", help="Data directory")
    parser.add_argument("--index", default="example_index.faiss", help="Index path")
    parser.add_argument("--query", default="What are the key principles of AI ethics?", help="Query to run")
    parser.add_argument("--device", default="cpu", help="Device for embedding model (cpu or cuda)")
    args = parser.parse_args()
    
    # Create data directory and example documents if needed
    data_dir = Path(args.data)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Creating example documents in {data_dir}...")
        create_example_documents(data_dir)
    
    # Initialize components
    adapter = TextAdapter(encoding="utf-8")
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="\n\n"
    )
    embedder = HuggingFaceEmbedder(
        model_name="Qwen/Qwen-Embedding",  # Use a smaller model if needed
        device=args.device,
        batch_size=4
    )
    indexer = FAISSIndexer(
        index_type="Flat",
        metric="cosine"
    )
    query_engine = TemplateQueryEngine(
        embedder=embedder,
        indexer=indexer,
        template="{query}"
    )
    
    # Extract text
    print("Extracting text...")
    texts = adapter.extract(data_dir)
    print(f"Extracted {len(texts)} documents")
    
    # Split text into chunks
    print("Splitting text into chunks...")
    chunks = splitter.split(texts)
    print(f"Created {len(chunks)} text chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    vectors = embedder.embed([chunk.text for chunk in chunks])
    print(f"Generated {len(vectors)} embedding vectors")
    
    # Build index
    print("Building index...")
    indexer.build(vectors, chunks)
    
    # Save index
    index_path = Path(args.index)
    print(f"Saving index to {index_path}...")
    indexer.save(index_path)
    
    # Process query
    query = args.query
    print(f"\nQuery: {query}")
    
    # Generate query embedding
    query_vectors = embedder.embed([query])
    query_vector = query_vectors[0]
    
    # Search index
    results = indexer.search(query_vector, top_k=3)
    
    # Print results
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Score: {result['score']:.4f}")
        
        # Print metadata if available
        if result.get("metadata"):
            meta_str = ", ".join([f"{k}: {v}" for k, v in result["metadata"].items() if k not in ["chunk_index", "total_chunks"]])
            if meta_str:
                print(f"Metadata: {meta_str}")
        
        # Print text
        print("---")
        print(result["text"])
        print("---")


if __name__ == "__main__":
    main() 