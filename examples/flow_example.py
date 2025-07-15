"""
Example demonstrating the usage of FlowController in TinySearch
"""
import os
from pathlib import Path

from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.flow.controller import FlowController


def main():
    """
    Main function demonstrating FlowController usage
    """
    # Create example directory and sample file if they don't exist
    data_dir = Path("./example_data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "sample.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("TinySearch is a lightweight vector retrieval system.\n")
            f.write("It is designed for easy integration and customization.\n")
            f.write("The system uses embeddings to search for relevant information.\n")
            f.write("You can use it for document retrieval and question answering.\n")
    
    # Create components
    adapter = TextAdapter()
    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    embedder = HuggingFaceEmbedder(model_name="Qwen/Qwen-Embedding", device="cpu")
    
    # Initialize FAISSIndexer
    # For this example, we know the dimension of Qwen-Embedding is 1536
    # In a real application, you'd want to dynamically determine this
    indexer = FAISSIndexer()  # FAISS will initialize dimension when building the index
    
    query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)
    
    # Configuration
    config = {
        "flow": {
            "use_cache": True,
            "cache_dir": ".cache"
        },
        "query_engine": {
            "top_k": 2
        }
    }
    
    # Create FlowController
    controller = FlowController(
        data_adapter=adapter,
        text_splitter=splitter,
        embedder=embedder,
        indexer=indexer,
        query_engine=query_engine,
        config=config
    )
    
    print("Building index...")
    controller.build_index(data_dir)
    
    # Save index for future use
    index_path = Path("./example_index.faiss")
    controller.save_index(index_path)
    print(f"Index saved to {index_path}")
    
    # Perform a query
    query = "What is TinySearch used for?"
    print(f"\nQuery: {query}")
    results = controller.query(query)
    
    print("\nResults:")
    for idx, result in enumerate(results):
        print(f"[{idx+1}] {result['chunk'].text} (Score: {result['score']:.4f})")
    
    # Show stats
    print("\nStats:")
    stats = controller.get_stats()
    print(f"Processed files: {stats['processed_files_count']}")
    print(f"Cache enabled: {stats['cache_enabled']}")
    
    # Clean up (optional)
    # controller.clear_cache()
    # if index_path.exists():
    #     index_path.unlink()


if __name__ == "__main__":
    main() 