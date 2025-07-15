#!/usr/bin/env python
"""
Advanced Features Demo for TinySearch

This example demonstrates the full capabilities of TinySearch with maximum customization,
showcasing:

1. Data validation utilities
2. Context window management
3. Response formatting in multiple formats
4. Hot-update capabilities
5. Advanced configuration options
6. Custom components
"""
import os
import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path
import yaml
import threading
import json
from typing import Dict, List, Any, Optional, Union

# install tinysearch (or locally)
# pip install tinysearch or pip install -e .

from tinysearch.config import Config
from tinysearch.validation import DataValidator, ValidationError
from tinysearch.context_window import ContextWindowManager
from tinysearch.formatters import get_formatter, ResponseFormatter
from tinysearch.base import DataAdapter, TextSplitter, Embedder, VectorIndexer, QueryEngine, TextChunk

# Import specific implementations
from tinysearch.adapters import TextAdapter, MarkdownAdapter, JSONAdapter, PDFAdapter
from tinysearch.splitters import CharacterTextSplitter
from tinysearch.embedders import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.flow.controller import FlowController

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom response formatter to demonstrate extensibility
class CustomFormatter(ResponseFormatter):
    """
    Custom response formatter that combines elements of Markdown and HTML
    """
    
    def __init__(self, highlight_terms: bool = True):
        self.highlight_terms = highlight_terms
        
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response using a custom format
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted response
        """
        if not results:
            return "# No Results Found\n\nYour search did not match any documents."
            
        query = kwargs.get("query", "")
        
        output = [
            f"# Search Results for '{query}'",
            f"Found {len(results)} matching documents.",
            ""
        ]
        
        for i, result in enumerate(results):
            output.append(f"## Match {i+1} (Score: {result.get('score', 0):.4f})")
            output.append("")
            
            # Get the text and highlight the query terms if enabled
            text = result.get("text", "")
            if self.highlight_terms and query:
                # Simple term highlighting using bold markdown
                for term in query.split():
                    if len(term) > 3:  # Only highlight meaningful terms
                        text = text.replace(term, f"**{term}**")
            
            output.append("```")
            output.append(text)
            output.append("```")
            output.append("")
            
            # Format metadata
            if "metadata" in result and result["metadata"]:
                output.append("### Source Information")
                output.append("<details>")
                output.append("<summary>Click to expand metadata</summary>")
                output.append("")
                output.append("| Property | Value |")
                output.append("| --- | --- |")
                
                for key, value in result["metadata"].items():
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    output.append(f"| {key} | {value} |")
                    
                output.append("")
                output.append("</details>")
                output.append("")
                
        return "\n".join(output)


# Custom reranking function for the query engine
def custom_reranker(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Custom reranking function that boosts results containing exact phrase matches
    
    Args:
        results: Original results from vector search
        query: Original query string
        
    Returns:
        Reranked results
    """
    if not results:
        return results
        
    # Simple exact phrase matching boost
    for result in results:
        text = result.get("text", "").lower()
        query_lower = query.lower()
        
        # Boost score for exact phrase match
        if query_lower in text:
            result["score"] = min(1.0, result.get("score", 0) * 1.2)
            result["match_type"] = "exact_phrase"
        else:
            result["match_type"] = "semantic"
    
    # Re-sort by score
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


# Create a temporary directory for our example
def setup_example_files(temp_dir: Path) -> None:
    """
    Set up example files for the demo
    
    Args:
        temp_dir: Directory to create example files in
    """
    # Create some text files with content
    files = {
        "document1.txt": """
        # Vector Search Systems
        
        Vector search is a technique used to find similar items in a dataset by comparing their vector representations.
        These vectors are typically generated using embedding models that convert text, images, or other data into 
        high-dimensional vectors.
        
        Key components of vector search systems include:
        
        1. Embedding generation
        2. Index building
        3. Similarity search
        4. Result ranking
        
        Vector search is particularly useful for semantic search, recommendation systems, and anomaly detection.
        """,
        
        "document2.md": """
        # Context Window Management
        
        Large Language Models (LLMs) have a limited context window that determines how much text they can process at once.
        Effective context window management is crucial for:
        
        - Ensuring the most relevant information is included
        - Optimizing token usage
        - Maintaining coherence in responses
        
        Techniques for context window management include truncation, sliding windows, and retrieval augmentation.
        """,
        
        "document3.json": json.dumps({
            "title": "Response Formatting Utilities",
            "content": "When building search systems, it's important to provide results in formats that are easy to consume. Response formatting utilities help convert raw search results into structured formats like JSON, Markdown, HTML, or custom formats. This makes integration with other systems much easier.",
            "examples": [
                "Plain text formatting",
                "Markdown for rich text applications",
                "JSON for API responses",
                "HTML for web interfaces"
            ]
        }),
        
        "document4.txt": """
        # Data Validation in Search Systems
        
        Data validation is an essential part of building robust search systems. It helps ensure that:
        
        1. Input data meets expected formats and constraints
        2. Embedding vectors have consistent dimensions
        3. Configuration parameters are valid
        4. Text chunks are properly formatted
        
        Good validation practices lead to more reliable search results and fewer runtime errors.
        """
    }
    
    for filename, content in files.items():
        file_path = temp_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
            
    # Create a directory to watch for hot updates
    watch_dir = temp_dir / "watch_dir"
    watch_dir.mkdir(exist_ok=True)
    
    # Create an initial file in the watch directory
    with open(watch_dir / "initial.txt", "w") as f:
        f.write("This is the initial file in the watch directory.")


# Main example function
def run_advanced_example():
    """Run the advanced features example"""
    logger.info("Starting TinySearch Advanced Features Demo")
    
    # Create a temporary directory for our example files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Set up example files
        setup_example_files(temp_dir)
        logger.info("Created example files")
        
        # 1. Data Validation - validate paths
        try:
            DataValidator.validate_directory_exists(temp_dir)
            logger.info("Validated temp directory exists")
            
            # Validate all expected files
            for filename in ["document1.txt", "document2.md", "document3.json", "document4.txt"]:
                DataValidator.validate_file_exists(temp_dir / filename)
            logger.info("All expected files exist")
            
            # Validate file extension
            DataValidator.validate_file_extension(temp_dir / "document1.txt", [".txt"])
            logger.info("Validated file extension")
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return
        
        # 2. Create configuration
        config_dict = {
            "adapter": {
                "type": "custom",  # We'll implement a custom adapter
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
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for demo
                "device": "cpu",  # Use CPU for the demo
                "normalize": True
            },
            "indexer": {
                "type": "faiss",
                "index_path": str(temp_dir / "index.faiss"),
                "metric": "cosine"
            },
            "query_engine": {
                "method": "template",
                "template": "Find information about: {query}",
                "top_k": 3
            },
            "flow": {
                "use_cache": True,
                "cache_dir": str(temp_dir / ".cache")
            }
        }
        
        # Validate configuration
        try:
            DataValidator.validate_config(config_dict, ["adapter", "splitter", "embedder", "indexer"])
            logger.info("Configuration validated")
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            return
            
        # Create configuration object
        config = Config()
        config.config = config_dict
        
        # 3. Create a custom adapter that can handle multiple file types
        class MultiFormatAdapter(DataAdapter):
            """Custom adapter that can handle multiple file formats"""
            
            def __init__(self, encoding: str = "utf-8"):
                self.encoding = encoding
                self.adapters = {
                    ".txt": TextAdapter(encoding=encoding),
                    ".md": MarkdownAdapter(encoding=encoding),
                    ".json": JSONAdapter(encoding=encoding, fields=["content"])
                }
                
            def extract(self, filepath: Union[str, Path]) -> List[str]:
                """Extract text from multiple file formats"""
                filepath = Path(filepath)
                extension = filepath.suffix.lower()
                
                if extension in self.adapters:
                    return self.adapters[extension].extract(filepath)
                else:
                    logger.warning(f"No adapter for extension {extension}, using text adapter")
                    return TextAdapter(encoding=self.encoding).extract(filepath)
        
        # 4. Initialize components
        data_adapter = MultiFormatAdapter(encoding="utf-8")
        text_splitter = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separator="\n\n"
        )
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            max_length=512,
            normalize_embeddings=True
        )
        
        # Get embedding dimension
        sample_text = "Sample text to determine embedding dimension."
        sample_embedding = embedder.embed([sample_text])[0]
        embedding_dimension = len(sample_embedding)
        logger.info(f"Detected embedding dimension: {embedding_dimension}")
        
        # Initialize the indexer with the correct dimension
        indexer = FAISSIndexer(
            metric="cosine",
            use_gpu=False  # Use CPU for the demo
        )
        # Store dimension for later use if needed
        indexer.dimension = embedding_dimension
        
        query_engine = TemplateQueryEngine(
            embedder=embedder,
            indexer=indexer,
            template="Find information about: {query}",
            rerank_fn=custom_reranker  # Use our custom reranker
        )
        
        # 5. Create flow controller
        flow_controller = FlowController(
            data_adapter=data_adapter,
            text_splitter=text_splitter,
            embedder=embedder,
            indexer=indexer,
            query_engine=query_engine,
            config=config_dict
        )
        
        # 6. Build index with validation
        logger.info("Building index...")
        try:
            flow_controller.build_index(temp_dir, extensions=[".txt", ".md", ".json"])
            logger.info("Index built successfully")
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return
            
        # 7. Set up hot-update watching on the watch directory
        watch_dir = temp_dir / "watch_dir"
        logger.info(f"Setting up hot-update on {watch_dir}")
        
        # Define a callback function for hot updates
        def hot_update_callback(updates, deletions):
            logger.info(f"Hot update received: {len(updates)} updates, {len(deletions)} deletions")
            
        # Start hot-update monitoring
        try:
            flow_controller.start_hot_update(
                watch_paths=[str(watch_dir)],
                file_extensions=[".txt", ".md", ".json"],
                process_delay=1.0,
                recursive=True,
                on_update_callback=hot_update_callback
            )
            logger.info("Hot-update monitoring started")
        except Exception as e:
            logger.error(f"Failed to start hot-update: {e}")
        
        # 8. Demonstrate context window management
        logger.info("Demonstrating context window management...")
        
        # Initialize context window manager
        context_manager = ContextWindowManager(
            max_tokens=1000,  # Small for demo purposes
            reserved_tokens=200,
            overlap_strategy="smart"
        )
        
        # Extract chunks to manage
        all_chunks = []
        all_metadata = []
        for file_path in (temp_dir / "document1.txt"), (temp_dir / "document4.txt"):
            texts = data_adapter.extract(file_path)
            metadata = [{"source": str(file_path)} for _ in range(len(texts))]
            chunks = text_splitter.split(texts, metadata)
            
            all_chunks.extend([chunk.text for chunk in chunks])
            all_metadata.extend([chunk.metadata for chunk in chunks])
        
        # Create context windows
        context_windows = context_manager.fit_text_to_window(all_chunks, all_metadata)
        logger.info(f"Created {len(context_windows)} context windows")
        
        # 9. Run some example queries with different formatters
        queries = [
            "What is vector search?",
            "How to manage context windows?",
            "Data validation techniques",
            "Response formatting options"
        ]
        
        formatters = {
            "text": get_formatter("text"),
            "markdown": get_formatter("markdown"),
            "json": get_formatter("json", pretty=True),
            "html": get_formatter("html"),
            "custom": CustomFormatter(highlight_terms=True)
        }
        
        for query in queries:
            logger.info(f"\nRunning query: {query}")
            results = flow_controller.query(query, top_k=2)
            
            # Format with each formatter
            for name, formatter in formatters.items():
                formatted = formatter.format_response(results, query=query)
                logger.info(f"\n{name.upper()} FORMAT EXAMPLE:")
                # Print just a preview for brevity
                preview = formatted.split("\n")[:5]
                preview.append("...")
                logger.info("\n".join(preview))
        
        # 10. Demonstrate hot-update capability
        logger.info("\nDemonstrating hot-update capability...")
        logger.info("Adding a new file to the watch directory")
        
        # Add a new file to the watch directory
        with open(watch_dir / "new_document.txt", "w") as f:
            f.write("""
            # Hot Update Feature
            
            Hot update is a powerful feature that allows the search system to automatically
            incorporate new documents and changes to existing documents without manual intervention.
            
            This capability is particularly useful for systems that need to stay up-to-date with
            rapidly changing content.
            """)
            
        # Wait a bit for the hot-update to process
        logger.info("Waiting for hot-update to process...")
        time.sleep(3)  # Give time for the file change to be detected and processed
        
        # Query for content in the new file
        logger.info("\nQuerying for content in the newly added file:")
        results = flow_controller.query("What is hot update?", top_k=1)
        
        # Format the results
        formatted = formatters["markdown"].format_response(results)
        logger.info(formatted)
        
        # Clean up and stop hot-update
        logger.info("\nStopping hot-update monitoring")
        try:
            flow_controller.stop_hot_update()  # type: ignore
        except Exception as e:
            logger.error(f"Failed to stop hot-update: {e}")
        
        logger.info("Advanced features demo completed successfully!")


if __name__ == "__main__":
    run_advanced_example() 