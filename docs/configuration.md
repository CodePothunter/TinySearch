# TinySearch Configuration Guide

TinySearch uses a flexible YAML-based configuration system that allows you to customize all aspects of the system. This guide explains the available configuration options and how to use them effectively.

## Configuration File Structure

A TinySearch configuration file is divided into sections corresponding to each major component:

```yaml
# Data adapter configuration
adapter:
  type: text
  params:
    # Adapter-specific parameters

# Text splitter configuration
splitter:
  type: character
  # Splitter-specific parameters

# Embedding model configuration
embedder:
  type: huggingface
  # Embedder-specific parameters

# Vector indexer configuration
indexer:
  type: faiss
  # Indexer-specific parameters

# Query engine configuration
query_engine:
  method: template
  # Query engine-specific parameters

# Flow controller configuration
flow:
  # Flow controller-specific parameters
```

## Complete Configuration Example

Here's a complete configuration file with all available options:

```yaml
# Data adapter configuration
adapter:
  type: text  # Options: text, pdf, csv, markdown, json, custom
  params:
    encoding: utf-8  # For text files
    # CSV-specific parameters
    # column: "text"
    # delimiter: ","
    # JSON-specific parameters
    # key_path: "content.text"
    # collection_key: "items"
    # Custom adapter parameters
    # module: "my_module"
    # class: "MyCustomAdapter"
    # init:
    #   custom_param1: value1
    #   custom_param2: value2

# Text splitter configuration
splitter:
  type: character  # Currently only "character" is supported
  chunk_size: 300  # Number of characters per chunk
  chunk_overlap: 50  # Overlap between chunks
  separator: "\n\n"  # Optional separator for chunking
  keep_separator: false  # Whether to keep the separator in chunks
  strip_whitespace: true  # Whether to strip whitespace from chunks

# Embedding model configuration
embedder:
  type: huggingface  # Currently only "huggingface" is supported
  model: "Qwen/Qwen-Embedding"  # HuggingFace model name
  device: "cuda"  # "cuda" or "cpu"
  max_length: 512  # Maximum sequence length
  batch_size: 8  # Batch size for embedding generation
  normalize: true  # Whether to normalize embeddings
  cache_dir: "~/.cache/tinysearch/models"  # Cache directory for models

# Vector indexer configuration
indexer:
  type: faiss  # Currently only "faiss" is supported
  index_type: "Flat"  # Options: "Flat", "IVF", "HNSW"
  metric: "cosine"  # Options: "cosine", "l2", "ip" (inner product)
  index_path: ".cache/index.faiss"  # Path to save/load the index (default: ./.cache/index.faiss)
  nlist: 100  # Number of clusters for IVF index
  nprobe: 10  # Number of clusters to search for IVF index
  use_gpu: false  # Whether to use GPU for indexing

# Query engine configuration
query_engine:
  method: template  # Currently only "template" is supported
  template: "Please help me find: {query}"  # Template for formatting queries
  top_k: 5  # Default number of results to return

# Flow controller configuration
flow:
  use_cache: true  # Whether to use caching
  cache_dir: ".cache"  # Cache directory
```

## Component-Specific Configurations

### Data Adapters

#### TextAdapter

```yaml
adapter:
  type: text
  params:
    encoding: utf-8  # File encoding
```

#### PDFAdapter

```yaml
adapter:
  type: pdf
  params:
    # No specific parameters for PDF adapter
```

#### CSVAdapter

```yaml
adapter:
  type: csv
  params:
    column: "text"  # Column containing text data
    encoding: "utf-8"  # File encoding
    delimiter: ","  # Column delimiter
```

#### MarkdownAdapter

```yaml
adapter:
  type: markdown
  params:
    # No specific parameters for markdown adapter
```

#### JSONAdapter

```yaml
adapter:
  type: json
  params:
    key_path: "content.text"  # Path to text field within JSON
    collection_key: "items"  # Optional path to list of items
```

#### Custom Adapter

```yaml
adapter:
  type: custom
  params:
    module: "my_module"  # Python module containing the adapter
    class: "MyCustomAdapter"  # Class name of the adapter
    init:  # Parameters to pass to the adapter's __init__
      param1: value1
      param2: value2
```

### Text Splitter

```yaml
splitter:
  type: character
  chunk_size: 300  # Number of characters per chunk
  chunk_overlap: 50  # Overlap between chunks
  separator: "\n\n"  # Optional separator (e.g., paragraphs)
  keep_separator: false  # Whether to keep separators in chunks
  strip_whitespace: true  # Whether to strip whitespace from chunks
```

### Embedder

```yaml
embedder:
  type: huggingface
  model: "Qwen/Qwen-Embedding"  # HuggingFace model name or path
  device: "cuda"  # "cuda" or "cpu" or specific device like "cuda:0"
  max_length: 512  # Maximum token length for the model
  batch_size: 8  # Batch size for efficiency
  normalize: true  # Whether to normalize embeddings to unit length
  cache_dir: "~/.cache/tinysearch/models"  # Cache directory for models
  # bf16: true  # (Optional) If set, will try bf16 inference on CPU. If not set, will auto-try bf16 on CPU and fallback to float32 if not supported.
```

> Note: When `device` is set to `cpu`, TinySearch will automatically attempt to use bf16 (bfloat16) inference for embedding generation if supported by your hardware and PyTorch version. If bf16 is not available, it will gracefully fall back to float32 and print a warning message.

### Vector Indexer

```yaml
indexer:
  type: faiss
  index_type: "Flat"  # "Flat" (exact search), "IVF" (approximate), "HNSW" (graph-based)
  metric: "cosine"  # "cosine", "l2" (Euclidean), "ip" (inner product)
  index_path: "index.faiss"  # Path to save/load index
  nlist: 100  # Number of clusters for IVF index (only for IVF)
  nprobe: 10  # Number of clusters to search (only for IVF)
  use_gpu: false  # Whether to use GPU acceleration
```

### Query Engine

```yaml
query_engine:
  method: template
  template: "Please help me find: {query}"  # Template with {query} placeholder
  top_k: 5  # Default number of results to return
```

### Flow Controller

```yaml
flow:
  use_cache: true  # Whether to use caching for processed files
  cache_dir: ".cache"  # Directory for cache storage
```

## Configuration Loading

TinySearch loads configuration in the following order of precedence:

1. **Default Configuration**: Built-in defaults
2. **Configuration File**: Settings from your configuration file
3. **Runtime Overrides**: Command-line arguments or API parameters

## Using Environment Variables

You can use environment variables in your configuration by using the `${ENV_VAR}` syntax:

```yaml
embedder:
  model: "${EMBEDDING_MODEL}"
  device: "${DEVICE:-cpu}"  # Default to "cpu" if DEVICE is not set
```

## Configuration Management API

You can also manage configuration programmatically:

```python
from tinysearch.config import Config

# Load configuration from file
config = Config("config.yaml")

# Access configuration values
model_name = config.config["embedder"]["model"]

# Modify configuration
config.config["embedder"]["device"] = "cpu"

# Save modified configuration
config.save("new_config.yaml")
```

## Best Practices

### For Small Documents

```yaml
splitter:
  chunk_size: 200
  chunk_overlap: 20

embedder:
  batch_size: 16
```

### For Large Documents

```yaml
splitter:
  chunk_size: 500
  chunk_overlap: 100

embedder:
  batch_size: 4
```

### For Speed

```yaml
indexer:
  index_type: "IVF"
  metric: "ip"  # Inner product is faster than cosine
  nlist: 100
  nprobe: 5  # Lower for faster search

flow:
  use_cache: true
```

### For Accuracy

```yaml
indexer:
  index_type: "Flat"  # Exact search
  metric: "cosine"  # Cosine is more robust for semantic similarity

splitter:
  chunk_size: 300
  chunk_overlap: 150  # Higher overlap preserves context
```

## Troubleshooting

### Configuration Validation

TinySearch validates your configuration when loading. If there are errors, check for:

1. **Invalid component types**: Ensure you're using supported component types
2. **Missing required parameters**: Check that all required parameters are provided
3. **Type mismatches**: Ensure parameter types match what's expected

### Common Issues

1. **Index not found**: Ensure `index_path` points to the correct location
2. **Model download errors**: Check `cache_dir` is writable and you have internet access
3. **Out of memory**: Reduce `batch_size` or `max_length` in the embedder configuration
4. **Slow performance**: Consider changing `index_type` or adjusting `chunk_size` 