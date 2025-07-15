"""
Command-line interface for TinySearch
"""
import argparse
import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type, cast

from .config import Config
from .base import DataAdapter, TextSplitter, Embedder, VectorIndexer, QueryEngine
from .adapters import TextAdapter, PDFAdapter, CSVAdapter, MarkdownAdapter, JSONAdapter
from .splitters import CharacterTextSplitter
from .embedders import HuggingFaceEmbedder
# 直接从模块导入
from .indexers.faiss_indexer import FAISSIndexer
from .query.template import TemplateQueryEngine


def load_adapter(config: Config) -> DataAdapter:
    """
    Load a data adapter based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        DataAdapter instance
    """
    adapter_type = config.get("adapter.type", "text")
    params = config.get("adapter.params", {})
    
    if adapter_type == "text":
        return TextAdapter(**params)
    elif adapter_type == "pdf":
        return PDFAdapter(**params)
    elif adapter_type == "csv":
        return CSVAdapter(**params)
    elif adapter_type == "markdown":
        return MarkdownAdapter(**params)
    elif adapter_type == "json":
        return JSONAdapter(**params)
    elif adapter_type == "custom":
        # Load a custom adapter class
        module_path = params.get("module", "")
        class_name = params.get("class", "")
        
        if not module_path or not class_name:
            raise ValueError("Custom adapter requires module and class parameters")
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(**params.get("init", {}))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load custom adapter: {e}")
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")


def load_splitter(config: Config) -> TextSplitter:
    """
    Load a text splitter based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        TextSplitter instance
    """
    splitter_type = config.get("splitter.type", "character")
    chunk_size = config.get("splitter.chunk_size", 300)
    chunk_overlap = config.get("splitter.chunk_overlap", 50)
    
    if splitter_type == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=config.get("splitter.separator", "\n\n"),
            keep_separator=config.get("splitter.keep_separator", False),
            strip_whitespace=config.get("splitter.strip_whitespace", True)
        )
    else:
        raise ValueError(f"Unsupported splitter type: {splitter_type}")


def load_embedder(config: Config) -> Embedder:
    """
    Load an embedder based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Embedder instance
    """
    embedder_type = config.get("embedder.type", "huggingface")
    
    if embedder_type == "huggingface":
        return HuggingFaceEmbedder(
            model_name=config.get("embedder.model", "Qwen/Qwen-Embedding"),
            device=config.get("embedder.device", None),
            max_length=config.get("embedder.max_length", 512),
            batch_size=config.get("embedder.batch_size", 8),
            normalize_embeddings=config.get("embedder.normalize", True),
            cache_dir=config.get("embedder.cache_dir", None)
        )
    else:
        raise ValueError(f"Unsupported embedder type: {embedder_type}")


def load_indexer(config: Config) -> FAISSIndexer:
    """
    Load an indexer based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        FAISSIndexer instance
    """
    indexer_type = config.get("indexer.type", "faiss")
    
    if indexer_type == "faiss":
        return FAISSIndexer(
            index_type=config.get("indexer.index_type", "Flat"),
            metric=config.get("indexer.metric", "cosine"),
            nlist=config.get("indexer.nlist", 100),
            nprobe=config.get("indexer.nprobe", 10),
            use_gpu=config.get("indexer.use_gpu", False)
        )
    else:
        raise ValueError(f"Unsupported indexer type: {indexer_type}")


def load_query_engine(config: Config, embedder: Embedder, indexer: FAISSIndexer) -> QueryEngine:
    """
    Load a query engine based on configuration
    
    Args:
        config: Configuration object
        embedder: Embedder instance
        indexer: FAISSIndexer instance
        
    Returns:
        QueryEngine instance
    """
    query_engine_type = config.get("query_engine.method", "template")
    
    if query_engine_type == "template":
        return TemplateQueryEngine(
            embedder=embedder,
            indexer=indexer,
            template=config.get("query_engine.template", "请帮我查找：{query}")
        )
    else:
        raise ValueError(f"Unsupported query engine type: {query_engine_type}")


def build_index(args: argparse.Namespace, config: Config) -> None:
    """
    Build a search index
    
    Args:
        args: Command-line arguments
        config: Configuration object
    """
    print(f"Building index from {args.data}...")
    
    # Load components
    adapter = load_adapter(config)
    splitter = load_splitter(config)
    embedder = load_embedder(config)
    indexer = load_indexer(config)
    
    # Extract text from data source
    print("Extracting text...")
    texts = adapter.extract(args.data)
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
    index_path = Path(config.get("indexer.index_path", "index.faiss"))
    print(f"Saving index to {index_path}...")
    indexer.save(index_path)
    
    print("Index built successfully")


def query_index(args: argparse.Namespace, config: Config) -> None:
    """
    Query a search index
    
    Args:
        args: Command-line arguments
        config: Configuration object
    """
    # Load components
    embedder = load_embedder(config)
    indexer = load_indexer(config)
    query_engine = load_query_engine(config, embedder, indexer)
    
    # Load index
    index_path = Path(config.get("indexer.index_path", "index.faiss"))
    print(f"Loading index from {index_path}...")
    indexer.load(index_path)
    
    # Process query
    query = args.q
    top_k = args.top_k or config.get("query_engine.top_k", 5)
    
    print(f"Query: {query}")
    print(f"Retrieving top {top_k} results...")
    
    # Execute query
    results = query_engine.retrieve(query, top_k=top_k)
    
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
    
    # Save results to file if requested
    if args.output:
        output_path = Path(args.output)
        print(f"Saving results to {output_path}...")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Main entry point for the CLI
    """
    parser = argparse.ArgumentParser(description="TinySearch: A lightweight vector retrieval system")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build a search index")
    index_parser.add_argument("--data", required=True, help="Path to data file or directory")
    index_parser.add_argument("--config", required=True, help="Path to configuration file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a search index")
    query_parser.add_argument("--q", required=True, help="Query string")
    query_parser.add_argument("--config", required=True, help="Path to configuration file")
    query_parser.add_argument("--top-k", type=int, help="Number of results to return")
    query_parser.add_argument("--output", help="Path to save results to (JSON format)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Execute command
    try:
        if args.command == "index":
            build_index(args, config)
        elif args.command == "query":
            query_index(args, config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 