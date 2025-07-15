# TinySearch User Guide

This guide provides comprehensive instructions for using TinySearch, a lightweight vector retrieval system designed for embedding, indexing, and searching over text data.

## Installation

### Basic Installation

```bash
pip install tinysearch
```

### Installation with Additional Features

```bash
# With API support
pip install tinysearch[api]

# With embedding models support
pip install tinysearch[embedders]

# With all document adapters
pip install tinysearch[adapters]

# With all features
pip install tinysearch[full]
```

## Core Concepts

TinySearch consists of several key components that work together to enable efficient text search:

1. **DataAdapter**: Extracts text from various file formats (TXT, PDF, CSV, Markdown, JSON)
2. **TextSplitter**: Chunks text into appropriate segments for embedding
3. **Embedder**: Generates vector embeddings from text chunks
4. **VectorIndexer**: Builds and maintains a FAISS index for efficient similarity search
5. **QueryEngine**: Processes queries and retrieves relevant context
6. **FlowController**: Orchestrates the entire data flow

## Configuration

TinySearch uses a YAML configuration file to control all aspects of the system. Here's a sample configuration:

```yaml
# Data adapter configuration
adapter:
  type: text  # Options: text, pdf, csv, markdown, json, custom
  params:
    encoding: utf-8

# Text splitter configuration
splitter:
  type: character
  chunk_size: 300  # Characters per chunk
  chunk_overlap: 50  # Overlap between chunks
  separator: "\n\n"  # Optional paragraph separator

# Embedding model configuration
embedder:
  type: huggingface
  model: Qwen/Qwen-Embedding  # Or any HuggingFace model
  device: cuda  # Set to "cpu" if no GPU is available
  normalize: true

# Vector indexer configuration
indexer:
  type: faiss
  index_path: index.faiss
  metric: cosine  # Options: cosine, l2, ip (inner product)
  index_type: Flat  # Options: Flat, IVF, HNSW

# Query engine configuration
query_engine:
  method: template
  template: "Please help me find: {query}"
  top_k: 5

# Flow controller configuration
flow:
  use_cache: true
  cache_dir: .cache
```

## Command Line Usage

### Indexing Documents

To build a search index from your documents:

```bash
tinysearch index --data ./your_documents --config config.yaml
```

Options:
- `--data`: Path to a file or directory containing documents
- `--config`: Path to your configuration file
- `--force`: Force reprocessing of all files, ignoring cache

### Querying

To search your indexed documents:

```bash
tinysearch query --q "Your search query" --config config.yaml --top-k 5
```

Options:
- `--q` or `--query`: Your search query
- `--config`: Path to your configuration file
- `--top-k`: Number of results to return (overrides config file)

### API Server

To start the API server:

```bash
tinysearch-api --config config.yaml --port 8000
```

Options:
- `--config`: Path to your configuration file
- `--port`: Port to run the server on (default: 8000)
- `--host`: Host to bind to (default: 127.0.0.1)

## Using the API

Once the API server is running, you can query it using HTTP requests:

### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your search query", "top_k": 5}'
```

Response format:

```json
{
  "results": [
    {
      "text": "The relevant text chunk",
      "score": 0.95,
      "metadata": {
        "source": "/path/to/original/file.txt"
      }
    },
    ...
  ]
}
```

### Index Building Endpoint

```bash
curl -X POST http://localhost:8000/build-index \
  -H "Content-Type: application/json" \
  -d '{"data_path": "./your_documents", "force_reprocess": false}'
```

## Advanced Usage

### Custom Data Adapters

You can create custom data adapters by implementing the `DataAdapter` interface:

```python
from tinysearch.base import DataAdapter

class MyCustomAdapter(DataAdapter):
    def __init__(self, special_param=None):
        self.special_param = special_param
        
    def extract(self, filepath):
        # Your code to extract text from the file
        # ...
        return [text1, text2, ...]
```

Then configure it in your `config.yaml`:

```yaml
adapter:
  type: custom
  params:
    module: my_module
    class: MyCustomAdapter
    init:
      special_param: value
```

### Using the FlowController Programmatically

You can use TinySearch directly in your Python code:

```python
from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.flow.controller import FlowController

# Create components
adapter = TextAdapter()
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedder = HuggingFaceEmbedder(model_name="Qwen/Qwen-Embedding", device="cpu")
indexer = FAISSIndexer()
query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)

# Configuration
config = {
    "flow": {
        "use_cache": True,
        "cache_dir": ".cache"
    },
    "query_engine": {
        "top_k": 5
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

# Build index
controller.build_index("./your_documents")

# Query
results = controller.query("Your search query")

# Process results
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['chunk'].text}")
    print(f"Source: {result['chunk'].metadata.get('source', 'Unknown')}")
    print("---")
```

## Web UI

TinySearch includes a simple web-based user interface that makes it easy to search your indexed documents and manage your index without using the command line.

### Starting the Web UI

The web UI is built into the API server. To start it, run:

```bash
tinysearch-api
```

By default, the server will start on `http://localhost:8000`. Open this URL in your web browser to access the UI.

### Using the Web Interface

The web interface consists of three main sections:

#### Search

The search tab allows you to query your index. Simply enter your search query and select how many results you'd like to see. The results will show:

- The text content of each matching chunk
- The source document
- The relevance score (higher is better)

#### Index Management

The Index Management tab provides tools to:

- **Upload Document**: Upload individual files to be indexed
- **Build Index**: Process a directory of files to build or update your index
- **Clear Index**: Remove all indexed documents and start fresh

#### Statistics

The Stats tab shows information about your current index, including:

- The number of processed files
- Whether caching is enabled
- A list of all processed files

### Customizing the Web UI

The web UI is built using Bootstrap 5 and vanilla JavaScript. If you'd like to customize its appearance or behavior, you can modify the files in the `tinysearch/api/static` directory.

## Troubleshooting

### Common Issues

1. **Model Download Failures**:
   - Ensure you have internet connectivity when using HuggingFace models for the first time
   - Set `cache_dir` in the embedder config to a writeable directory

2. **Out of Memory Errors**:
   - Reduce `batch_size` in the embedder configuration
   - Use a smaller embedding model
   - Process smaller datasets or reduce chunk size

3. **Slow Indexing**:
   - Enable caching with `use_cache: true` in the flow configuration
   - Use a faster FAISS index type (like IVF), but note this may reduce accuracy

4. **Poor Search Results**:
   - Adjust chunk size and overlap to better match your content
   - Use a more appropriate embedding model for your domain
   - Try different similarity metrics (cosine, L2, inner product)

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/yourusername/tinysearch/docs)
2. Open an issue on [GitHub](https://github.com/yourusername/tinysearch/issues)

## License

TinySearch is licensed under the MIT License. 