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
from .base import (
    DataAdapter, TextSplitter, Embedder, VectorIndexer, QueryEngine,
    Retriever, FusionStrategy, Reranker,
)
from .adapters import TextAdapter, PDFAdapter, CSVAdapter, MarkdownAdapter, JSONAdapter
from .splitters import CharacterTextSplitter
from .embedders import HuggingFaceEmbedder
# 直接从模块导入
from .indexers.faiss_indexer import FAISSIndexer
from .query.template import TemplateQueryEngine
from .query.hybrid import HybridQueryEngine
from .retrievers.vector_retriever import VectorRetriever
from .retrievers.bm25_retriever import BM25Retriever
from .retrievers.substring_retriever import SubstringRetriever
from .fusion.rrf import ReciprocalRankFusion
from .fusion.weighted import WeightedFusion
from .rerankers.cross_encoder import CrossEncoderReranker
from .logger import get_logger, configure_logger, log_step, log_progress, log_success, log_error


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


def load_retriever(config: Config, retriever_config: Dict[str, Any], embedder: Embedder, indexer: FAISSIndexer) -> Retriever:
    """
    Load a single retriever based on its config dict.

    Args:
        config: Global configuration object
        retriever_config: Dict like {"type": "vector"} or {"type": "bm25", "tokenizer": "jieba"}
        embedder: Embedder instance (used by vector retriever)
        indexer: FAISSIndexer instance (used by vector retriever)

    Returns:
        Retriever instance
    """
    rtype = retriever_config.get("type", "vector")

    if rtype == "vector":
        return VectorRetriever(
            embedder=embedder,
            indexer=indexer,
            query_template=config.get("query_engine.template"),
        )
    elif rtype == "bm25":
        tokenizer_name = retriever_config.get("tokenizer", "default")
        tokenizer = None  # use default
        if tokenizer_name == "jieba":
            try:
                import jieba
                tokenizer = lambda text: list(jieba.cut(text.lower()))
            except ImportError:
                pass  # fallback to default
        return BM25Retriever(tokenizer=tokenizer)
    elif rtype == "substring":
        return SubstringRetriever(
            is_regex=retriever_config.get("is_regex", False),
        )
    else:
        raise ValueError(f"Unsupported retriever type: {rtype}")


def load_retrievers(config: Config, embedder: Embedder, indexer: FAISSIndexer) -> List[Retriever]:
    """
    Load all configured retrievers.

    Returns:
        List of Retriever instances
    """
    retriever_configs = config.get("retrievers", [{"type": "vector"}])
    return [
        load_retriever(config, rc, embedder, indexer)
        for rc in retriever_configs
    ]


def load_fusion(config: Config) -> FusionStrategy:
    """
    Load a fusion strategy based on configuration.

    Returns:
        FusionStrategy instance
    """
    strategy = config.get("fusion.strategy", "weighted")
    if strategy == "rrf":
        return ReciprocalRankFusion(
            k=config.get("fusion.k", 60),
        )
    elif strategy == "weighted":
        return WeightedFusion(
            weights=config.get("fusion.weights"),
            min_score=config.get("fusion.min_score", 0.0),
        )
    else:
        raise ValueError(f"Unsupported fusion strategy: {strategy}")


def load_reranker(config: Config) -> Optional[Reranker]:
    """
    Load a reranker if enabled in configuration.

    Returns:
        Reranker instance or None
    """
    if not config.get("reranker.enabled", False):
        return None

    return CrossEncoderReranker(
        model_name=config.get("reranker.model", "BAAI/bge-reranker-v2-m3"),
        device=config.get("reranker.device"),
        batch_size=config.get("reranker.batch_size", 64),
        max_length=config.get("reranker.max_length", 512),
        use_fp16=config.get("reranker.use_fp16", True),
    )


def load_query_engine(config: Config, embedder: Embedder, indexer: FAISSIndexer) -> QueryEngine:
    """
    Load a query engine based on configuration.

    Supports:
    - "template": Original TemplateQueryEngine (backward compatible)
    - "hybrid": HybridQueryEngine with multi-retriever fusion

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
    elif query_engine_type == "hybrid":
        retrievers = load_retrievers(config, embedder, indexer)
        fusion = load_fusion(config)
        reranker = load_reranker(config)
        return HybridQueryEngine(
            retrievers=retrievers,
            fusion_strategy=fusion,
            reranker=reranker,
            recall_multiplier=config.get("query_engine.recall_multiplier", 2),
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
    logger = get_logger("build_index")
    logger.info(f"🚀 Building index from {args.data}")

    # Load components
    adapter = load_adapter(config)
    splitter = load_splitter(config)
    embedder = load_embedder(config)
    indexer = load_indexer(config)

    # Extract text from data source
    log_step("Extracting text")
    texts = adapter.extract(args.data)
    logger.info(f"📄 Extracted {len(texts)} documents")

    # Split text into chunks
    log_step("Splitting text into chunks")
    chunks = splitter.split(texts)
    logger.info(f"✂️  Created {len(chunks)} text chunks")

    # Generate embeddings
    log_step("Generating embeddings")
    vectors = embedder.embed([chunk.text for chunk in chunks])
    logger.info(f"🧠 Generated {len(vectors)} embedding vectors")

    # Build index
    log_step("Building index")
    indexer.build(vectors, chunks)

    # Save index
    index_path = Path(config.get("indexer.index_path", "index.faiss"))
    log_step(f"Saving index to {index_path}")
    indexer.save(index_path)

    log_success("Index built successfully")


def query_index(args: argparse.Namespace, config: Config) -> None:
    """
    Query a search index

    Args:
        args: Command-line arguments
        config: Configuration object
    """
    logger = get_logger("query_index")

    # Load components
    embedder = load_embedder(config)
    indexer = load_indexer(config)
    query_engine = load_query_engine(config, embedder, indexer)

    # Load index
    index_path = Path(config.get("indexer.index_path", "index.faiss"))
    log_step(f"Loading index from {index_path}")
    indexer.load(index_path)

    # Process query
    query = args.q
    top_k = args.top_k or config.get("query_engine.top_k", 5)

    logger.info(f"🔍 Query: {query}")
    logger.info(f"📊 Retrieving top {top_k} results")

    # Execute query
    results = query_engine.retrieve(query, top_k=top_k)

    # Print results
    logger.info("\n📋 Results:")
    for i, result in enumerate(results):
        logger.info(f"\n[{i+1}] Score: {result['score']:.4f}")

        # Print metadata if available
        if result.get("metadata"):
            meta_str = ", ".join([f"{k}: {v}" for k, v in result["metadata"].items() if k not in ["chunk_index", "total_chunks"]])
            if meta_str:
                logger.info(f"📝 Metadata: {meta_str}")

        # Print text
        logger.info("---")
        logger.info(result["text"])
        logger.info("---")

    # Save results to file if requested
    if args.output:
        output_path = Path(args.output)
        log_step(f"Saving results to {output_path}")

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

    # Load configuration and configure logger
    try:
        config = Config(args.config)
        configure_logger(config.data)
        logger = get_logger("main")
    except Exception as e:
        log_error(f"Error loading configuration: {e}")
        return

    # Execute command
    try:
        if args.command == "index":
            build_index(args, config)
        elif args.command == "query":
            query_index(args, config)
    except Exception as e:
        import traceback
        log_error(f"Command execution failed: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main() 