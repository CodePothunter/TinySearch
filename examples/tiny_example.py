"""
TinySearch - 10-line vector search system

This example demonstrates how to build a complete vector search system
in just 10 lines of code using TinySearch's core components.
"""

from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer


def main():
    """Build a vector search system in 10 lines"""
    print("TinySearch: 10-Line Vector Search System")
    print("=======================================")
    
    # 1. Load documents - one line
    texts = TextAdapter().extract("example_data/sample.txt")
    
    # 2. Split text into chunks - one line
    chunks = CharacterTextSplitter(chunk_size=20, chunk_overlap=10).split(texts)
    
    # 3. Generate embeddings - one line  
    vectors = HuggingFaceEmbedder(device="cpu").embed([chunk.text for chunk in chunks])
    
    # 4. Build search index - one line
    indexer = FAISSIndexer()
    indexer.build(vectors, chunks)
    
    # 5. Search! - one line
    results = indexer.search(HuggingFaceEmbedder(device="cpu").embed(["What is TinySearch?"])[0], top_k=3)
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. [{result['score']:.4f}] {result['text'][:100]}...")


if __name__ == "__main__":
    main()