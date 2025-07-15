# TinySearch Feature Summary

This document provides a summary of the recently implemented features in TinySearch.

## Data Validation Utilities

TinySearch now includes comprehensive data validation utilities in the `tinysearch.validation` module. These utilities help ensure data integrity throughout the processing pipeline:

- **File and Directory Validation**: Verify that files and directories exist and have the correct format.
- **Embedding Validation**: Check that embeddings have consistent dimensions and proper numeric values.
- **Configuration Validation**: Validate configuration dictionaries against required keys and schemas.
- **Text and List Validation**: Ensure that text strings and lists are not empty.
- **Custom Validation**: Support for custom validation functions with clear error messages.

Example usage:

```python
from tinysearch.validation import DataValidator, ValidationError

# Validate a file exists
try:
    file_path = DataValidator.validate_file_exists("data/documents.txt")
    print(f"Valid file: {file_path}")
except ValidationError as e:
    print(f"Validation error: {e}")

# Validate embeddings
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
try:
    DataValidator.validate_embeddings(embeddings, expected_dim=3)
    print("Valid embeddings")
except ValidationError as e:
    print(f"Embedding validation error: {e}")
```

## Context Window Management

TinySearch now supports context window management through the `tinysearch.context_window` module. This feature helps optimize content for LLM processing:

- **Token Counting**: Estimate token counts for text chunks.
- **Window Sizing**: Fit text chunks into context windows with token limits.
- **Window Merging**: Merge multiple context windows with configurable overlap strategies.
- **Query-Specific Contexts**: Generate optimal context windows for specific queries.

Example usage:

```python
from tinysearch.context_window import ContextWindowManager

# Initialize context window manager
manager = ContextWindowManager(
    max_tokens=4096,
    reserved_tokens=1000,  # For prompts and responses
    overlap_strategy="smart"
)

# Fit text chunks into windows
text_chunks = ["Long chunk of text...", "Another chunk of text..."]
metadata_list = [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
windows = manager.fit_text_to_window(text_chunks, metadata_list)

# Generate context for a specific query
query = "What is vector search?"
context_text, context_metadata = manager.generate_context_for_query(
    query, text_chunks, metadata_list
)
```

## Response Formatting Utilities

TinySearch now includes response formatting utilities in the `tinysearch.formatters` module, providing multiple output formats for search results:

- **Plain Text**: Simple text formatting with configurable separators.
- **Markdown**: Rich text formatting with headers, code blocks, and metadata sections.
- **JSON**: Structured data format with optional pretty-printing and timestamps.
- **HTML**: Web-ready formatting with built-in CSS styling.

Example usage:

```python
from tinysearch.formatters import get_formatter

# Get search results from query engine
results = query_engine.retrieve("quantum computing", top_k=3)

# Format as plain text
text_formatter = get_formatter("text", include_scores=True)
text_output = text_formatter.format_response(results)
print(text_output)

# Format as Markdown
md_formatter = get_formatter("markdown", link_sources=True)
md_output = md_formatter.format_response(results)
print(md_output)

# Format as JSON
json_formatter = get_formatter("json", pretty=True)
json_output = json_formatter.format_response(results)
print(json_output)
```

## Hot-Update Capabilities

TinySearch now has integrated hot-update capabilities in the `tinysearch.flow` module, enabling real-time index updates when source documents change:

- **File Change Detection**: Automatically detect when files are created, modified, deleted, or moved.
- **Delayed Processing**: Group changes together to avoid redundant processing.
- **Selective Watching**: Monitor specific file extensions or directories.
- **Recursive Monitoring**: Watch subdirectories for changes.
- **Update Callbacks**: Register callbacks for custom actions after updates.

Example usage:

```python
from tinysearch.flow import FlowController

# Initialize components and flow controller
flow_controller = FlowController(
    data_adapter=data_adapter,
    text_splitter=text_splitter,
    embedder=embedder,
    indexer=indexer,
    query_engine=query_engine,
    config=config_dict
)

# Start hot update monitoring
flow_controller.start_hot_update(
    watch_paths=["data/documents"],
    file_extensions=[".txt", ".md", ".pdf"],
    process_delay=1.0,
    recursive=True,
    on_update_callback=lambda updates, deletions: print(f"Updated: {len(updates)}, Deleted: {len(deletions)}")
)

# Later, stop hot update monitoring when needed
flow_controller.stop_hot_update()
```

## Next Steps

With these features implemented, TinySearch is now a more robust and flexible vector retrieval system. Future development will focus on:

1. Docker deployment for simplified installation
2. Example configurations for common use cases
3. Performance monitoring and optimization
4. Advanced reranking for improved search results
5. Hybrid search capabilities combining vector and keyword-based approaches 