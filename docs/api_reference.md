# TinySearch API Reference

This document provides detailed information about TinySearch's core APIs and components.

## Core Interfaces

All TinySearch components implement abstract base classes that define their interfaces. These are defined in `tinysearch.base`.

### DataAdapter

```python
class DataAdapter(ABC):
    """Interface for adapters that extract text from different data formats."""
    
    @abstractmethod
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        Extract text content from the given file
        
        Args:
            filepath: Path to the file to extract text from
            
        Returns:
            List of text strings extracted from the file
        """
        pass
```

### TextSplitter

```python
class TextSplitter(ABC):
    """Interface for text splitters that chunk text into smaller segments"""
    
    @abstractmethod
    def split(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        Split texts into chunks
        
        Args:
            texts: List of text strings to split
            metadata: Optional list of metadata dicts corresponding to each text
            
        Returns:
            List of TextChunk objects
        """
        pass
```

### Embedder

```python
class Embedder(ABC):
    """Interface for embedding models that convert text to vectors"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to embedding vectors
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as float lists
        """
        pass
```

### VectorIndexer

```python
class VectorIndexer(ABC):
    """Interface for vector indexers that build and maintain search indices"""
    
    @abstractmethod
    def build(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """
        Build the index from vectors and their corresponding text chunks
        
        Args:
            vectors: List of embedding vectors
            texts: List of TextChunk objects corresponding to the vectors
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for vectors similar to the query vector
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the index to disk
        
        Args:
            path: Path to save the index to
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the index from disk
        
        Args:
            path: Path to load the index from
        """
        pass
```

### QueryEngine

```python
class QueryEngine(ABC):
    """Interface for query engines that process user queries"""
    
    @abstractmethod
    def format_query(self, query: str) -> str:
        """
        Format the raw query string
        
        Args:
            query: Raw query string
            
        Returns:
            Formatted query string
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        pass
```

### FlowController

```python
class FlowController(ABC):
    """Interface for flow controllers that orchestrate the data pipeline"""
    
    @abstractmethod
    def build_index(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        Build the search index from data files
        
        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional arguments for customizing the build process
        """
        pass
    
    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a query and return relevant chunks
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments for customizing the query process
            
        Returns:
            List of dictionaries containing text chunks and similarity scores
        """
        pass
    
    @abstractmethod
    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the built index to disk
        
        Args:
            path: Path to save the index to, if None use a default path
        """
        pass
    
    @abstractmethod
    def load_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load an index from disk
        
        Args:
            path: Path to load the index from, if None use a default path
        """
        pass
```

## Data Models

### TextChunk

```python
class TextChunk:
    """Represents a chunk of text with optional metadata"""
    
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}
```

## Implementation Details

### DataAdapter Implementations

#### TextAdapter

```python
from tinysearch.adapters.text import TextAdapter

# Initialize with default settings
adapter = TextAdapter(encoding="utf-8")

# Extract text from file
texts = adapter.extract("path/to/file.txt")
```

#### PDFAdapter

```python
from tinysearch.adapters.pdf import PDFAdapter

# Initialize with default settings
adapter = PDFAdapter()

# Extract text from file
texts = adapter.extract("path/to/file.pdf")
```

#### CSVAdapter

```python
from tinysearch.adapters.csv import CSVAdapter

# Initialize with custom settings
adapter = CSVAdapter(
    column="text_column",  # Column to extract text from
    encoding="utf-8",
    delimiter=","
)

# Extract text from file
texts = adapter.extract("path/to/file.csv")
```

#### MarkdownAdapter

```python
from tinysearch.adapters.markdown import MarkdownAdapter

# Initialize with default settings
adapter = MarkdownAdapter()

# Extract text from file
texts = adapter.extract("path/to/file.md")
```

#### JSONAdapter

```python
from tinysearch.adapters.json_adapter import JSONAdapter

# Initialize with custom settings
adapter = JSONAdapter(
    key_path="content.text",  # Path to text within JSON structure
    collection_key="items"    # Optional path to list of items
)

# Extract text from file
texts = adapter.extract("path/to/file.json")
```

### TextSplitter Implementations

#### CharacterTextSplitter

```python
from tinysearch.splitters.character import CharacterTextSplitter

# Initialize with custom settings
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n\n",
    keep_separator=False,
    strip_whitespace=True
)

# Split text into chunks
chunks = splitter.split(["Your text content here"])
```

### Embedder Implementations

#### HuggingFaceEmbedder

```python
from tinysearch.embedders.huggingface import HuggingFaceEmbedder

# Initialize with custom settings
embedder = HuggingFaceEmbedder(
    model_name="Qwen/Qwen-Embedding",
    device="cuda",  # or "cpu"
    max_length=512,
    batch_size=8,
    normalize_embeddings=True,
    cache_dir=None
)

# Generate embeddings
vectors = embedder.embed(["Your text content here"])
```

### VectorIndexer Implementations

#### FAISSIndexer

```python
from tinysearch.indexers.faiss_indexer import FAISSIndexer

# Initialize with custom settings
indexer = FAISSIndexer(
    index_type="Flat",  # Options: "Flat", "IVF", "HNSW"
    metric="cosine",    # Options: "cosine", "l2", "ip"
    nlist=100,          # For IVF index
    nprobe=10,          # For IVF index search
    use_gpu=False       # Whether to use GPU
)

# Build index
indexer.build(vectors, chunks)

# Search index
results = indexer.search(query_vector, top_k=5)

# Save/load index
indexer.save("path/to/index.faiss")
indexer.load("path/to/index.faiss")
```

### QueryEngine Implementations

#### TemplateQueryEngine

```python
from tinysearch.query.template import TemplateQueryEngine

# Initialize with custom settings
query_engine = TemplateQueryEngine(
    embedder=embedder,
    indexer=indexer,
    template="Please help me find: {query}",
    rerank_fn=None  # Optional function to rerank results
)

# Format query
formatted_query = query_engine.format_query("original query")

# Retrieve results
results = query_engine.retrieve("your query", top_k=5)
```

### FlowController Implementation

```python
from tinysearch.flow.controller import FlowController

# Initialize with all components
controller = FlowController(
    data_adapter=adapter,
    text_splitter=splitter,
    embedder=embedder,
    indexer=indexer,
    query_engine=query_engine,
    config={
        "flow": {
            "use_cache": True,
            "cache_dir": ".cache"
        },
        "query_engine": {
            "top_k": 5
        }
    }
)

# Build index
controller.build_index("./your_documents", force_reprocess=False)

# Save/load index
controller.save_index("path/to/index.faiss")
controller.load_index("path/to/index.faiss")

# Query
results = controller.query("your query", top_k=5)

# Clear cache
controller.clear_cache()

# Get stats
stats = controller.get_stats()
```

## Configuration Management

TinySearch provides a configuration management system:

```python
from tinysearch.config import Config

# Load config from file
config = Config("config.yaml")

# Access config values
adapter_type = config.config["adapter"]["type"]

# Set config values
config.config["embedder"]["device"] = "cpu"

# Save config to file
config.save("new_config.yaml")
```

## Command Line Interface

TinySearch provides a command line interface via the `tinysearch` command:

```bash
# Index documents
tinysearch index --data ./your_documents --config config.yaml

# Query
tinysearch query --q "Your search query" --config config.yaml

# Start API server
tinysearch-api --config config.yaml --port 8000
```

## REST API

When running the API server, the following endpoints are available:

### POST /query

Query the index:

```json
{
  "query": "Your search query",
  "top_k": 5
}
```

Response:

```json
{
  "results": [
    {
      "text": "The relevant text chunk",
      "score": 0.95,
      "metadata": {
        "source": "/path/to/original/file.txt"
      }
    }
  ]
}
```

### POST /build-index

Build or update the index:

```json
{
  "data_path": "./your_documents",
  "force_reprocess": false
}
```

Response:

```json
{
  "status": "success",
  "message": "Index built successfully",
  "files_processed": 10
}
```

### GET /stats

Get system statistics:

```json
{
  "processed_files_count": 10,
  "cache_enabled": true,
  "cache_directory": ".cache",
  "config": {
    "adapter": {
      "type": "text"
    },
    // ...other config
  }
}
``` 