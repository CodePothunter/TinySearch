# TinySearch API 参考

本文档提供了关于 TinySearch 核心 API 和组件的详细信息。

## 核心接口

所有 TinySearch 组件都实现了定义其接口的抽象基类。这些接口定义在 `tinysearch.base` 中。

### DataAdapter（数据适配器）

```python
class DataAdapter(ABC):
    """提取不同数据格式文本的适配器接口。"""
    
    @abstractmethod
    def extract(self, filepath: Union[str, Path]) -> List[str]:
        """
        从给定文件中提取文本内容
        
        参数:
            filepath: 要提取文本的文件路径
            
        返回:
            从文件中提取的文本字符串列表
        """
        pass
```

### TextSplitter（文本分割器）

```python
class TextSplitter(ABC):
    """将文本分割成更小段落的分割器接口"""
    
    @abstractmethod
    def split(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        将文本分割成块
        
        参数:
            texts: 要分割的文本字符串列表
            metadata: 可选的与每个文本对应的元数据字典列表
            
        返回:
            TextChunk 对象列表
        """
        pass
```

### Embedder（嵌入器）

```python
class Embedder(ABC):
    """将文本转换为向量的嵌入模型接口"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本转换为嵌入向量
        
        参数:
            texts: 要嵌入的文本字符串列表
            
        返回:
            嵌入向量列表（浮点数列表）
        """
        pass
```

### VectorIndexer（向量索引器）

```python
class VectorIndexer(ABC):
    """构建和维护搜索索引的向量索引器接口"""
    
    @abstractmethod
    def build(self, vectors: List[List[float]], texts: List[TextChunk]) -> None:
        """
        从向量及其对应的文本块构建索引
        
        参数:
            vectors: 嵌入向量列表
            texts: 与向量对应的 TextChunk 对象列表
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在索引中搜索与查询向量相似的向量
        
        参数:
            query_vector: 查询嵌入向量
            top_k: 返回结果的数量
            
        返回:
            包含文本块和相似度分数的字典列表
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        将索引保存到磁盘
        
        参数:
            path: 保存索引的路径
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        从磁盘加载索引
        
        参数:
            path: 加载索引的路径
        """
        pass
```

### QueryEngine（查询引擎）

```python
class QueryEngine(ABC):
    """处理用户查询的查询引擎接口"""
    
    @abstractmethod
    def format_query(self, query: str) -> str:
        """
        格式化原始查询字符串
        
        参数:
            query: 原始查询字符串
            
        返回:
            格式化后的查询字符串
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索与查询相关的文本块
        
        参数:
            query: 查询字符串
            top_k: 返回结果的数量
            
        返回:
            包含文本块和相似度分数的字典列表
        """
        pass
```

### FlowController（流程控制器）

```python
class FlowController(ABC):
    """编排数据流水线的流程控制器接口"""
    
    @abstractmethod
    def build_index(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        从数据文件构建搜索索引
        
        参数:
            data_path: 数据文件或目录的路径
            **kwargs: 用于自定义构建过程的附加参数
        """
        pass
    
    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        处理查询并返回相关文本块
        
        参数:
            query_text: 查询字符串
            top_k: 返回结果的数量
            **kwargs: 用于自定义查询过程的附加参数
            
        返回:
            包含文本块和相似度分数的字典列表
        """
        pass
    
    @abstractmethod
    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        将构建的索引保存到磁盘
        
        参数:
            path: 保存索引的路径，如果为 None，则使用默认路径
        """
        pass
    
    @abstractmethod
    def load_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        从磁盘加载索引
        
        参数:
            path: 加载索引的路径，如果为 None，则使用默认路径
        """
        pass
```

## 数据模型

### TextChunk（文本块）

```python
class TextChunk:
    """表示带有可选元数据的文本块"""
    
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}
```

## 实现细节

### DataAdapter 实现

#### TextAdapter（文本适配器）

```python
from tinysearch.adapters.text import TextAdapter

# 使用默认设置初始化
adapter = TextAdapter(encoding="utf-8")

# 从文件提取文本
texts = adapter.extract("path/to/file.txt")
```

#### PDFAdapter（PDF适配器）

```python
from tinysearch.adapters.pdf import PDFAdapter

# 使用默认设置初始化
adapter = PDFAdapter()

# 从文件提取文本
texts = adapter.extract("path/to/file.pdf")
```

#### CSVAdapter（CSV适配器）

```python
from tinysearch.adapters.csv import CSVAdapter

# 使用自定义设置初始化
adapter = CSVAdapter(
    column="text_column",  # 要提取文本的列
    encoding="utf-8",
    delimiter=","
)

# 从文件提取文本
texts = adapter.extract("path/to/file.csv")
```

#### MarkdownAdapter（Markdown适配器）

```python
from tinysearch.adapters.markdown import MarkdownAdapter

# 使用默认设置初始化
adapter = MarkdownAdapter()

# 从文件提取文本
texts = adapter.extract("path/to/file.md")
```

#### JSONAdapter（JSON适配器）

```python
from tinysearch.adapters.json_adapter import JSONAdapter

# 使用自定义设置初始化
adapter = JSONAdapter(
    key_path="content.text",  # JSON结构中文本的路径
    collection_key="items"    # 可选的项目列表路径
)

# 从文件提取文本
texts = adapter.extract("path/to/file.json")
```

### TextSplitter 实现

#### CharacterTextSplitter（字符文本分割器）

```python
from tinysearch.splitters.character import CharacterTextSplitter

# 使用自定义设置初始化
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separator="\n\n",
    keep_separator=False,
    strip_whitespace=True
)

# 将文本分割成块
chunks = splitter.split(["您的文本内容"])
```

### Embedder 实现

#### HuggingFaceEmbedder（HuggingFace嵌入器）

```python
from tinysearch.embedders.huggingface import HuggingFaceEmbedder

# 使用自定义设置初始化
embedder = HuggingFaceEmbedder(
    model_name="Qwen/Qwen-Embedding",
    device="cuda",  # 或 "cpu"
    max_length=512,
    batch_size=8,
    normalize_embeddings=True,
    cache_dir=None
)

# 生成嵌入
vectors = embedder.embed(["您的文本内容"])
```

### VectorIndexer 实现

#### FAISSIndexer（FAISS索引器）

```python
from tinysearch.indexers.faiss_indexer import FAISSIndexer

# 使用自定义设置初始化
indexer = FAISSIndexer(
    index_type="Flat",  # 选项: "Flat", "IVF", "HNSW"
    metric="cosine",    # 选项: "cosine", "l2", "ip"
    nlist=100,          # 用于 IVF 索引
    nprobe=10,          # 用于 IVF 索引搜索
    use_gpu=False       # 是否使用 GPU
)

# 构建索引
indexer.build(vectors, chunks)

# 搜索索引
results = indexer.search(query_vector, top_k=5)

# 保存/加载索引
indexer.save("path/to/index.faiss")
indexer.load("path/to/index.faiss")
```

### QueryEngine 实现

#### TemplateQueryEngine（模板查询引擎）

```python
from tinysearch.query.template import TemplateQueryEngine

# 使用自定义设置初始化
query_engine = TemplateQueryEngine(
    embedder=embedder,
    indexer=indexer,
    template="请帮我查找：{query}",
    rerank_fn=None  # 可选的重排序函数
)

# 格式化查询
formatted_query = query_engine.format_query("原始查询")

# 检索结果
results = query_engine.retrieve("您的查询", top_k=5)
```

### FlowController 实现

```python
from tinysearch.flow.controller import FlowController

# 使用所有组件初始化
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

# 构建索引
controller.build_index("./your_documents", force_reprocess=False)

# 保存/加载索引
controller.save_index("path/to/index.faiss")
controller.load_index("path/to/index.faiss")

# 查询
results = controller.query("您的查询", top_k=5)

# 清除缓存
controller.clear_cache()

# 获取统计信息
stats = controller.get_stats()
```

## 配置管理

TinySearch 提供了配置管理系统：

```python
from tinysearch.config import Config

# 从文件加载配置
config = Config("config.yaml")

# 访问配置值
adapter_type = config.config["adapter"]["type"]

# 设置配置值
config.config["embedder"]["device"] = "cpu"

# 保存配置到文件
config.save("new_config.yaml")
```

## 命令行界面

TinySearch 通过 `tinysearch` 命令提供命令行界面：

```bash
# 索引文档
tinysearch index --data ./your_documents --config config.yaml

# 查询
tinysearch query --q "您的搜索查询" --config config.yaml

# 启动 API 服务器
tinysearch-api --config config.yaml --port 8000
```

## REST API

运行 API 服务器时，提供以下端点：

### POST /query

查询索引：

```json
{
  "query": "您的搜索查询",
  "top_k": 5
}
```

响应：

```json
{
  "results": [
    {
      "text": "相关的文本块",
      "score": 0.95,
      "metadata": {
        "source": "/path/to/original/file.txt"
      }
    }
  ]
}
```

### POST /build-index

构建或更新索引：

```json
{
  "data_path": "./your_documents",
  "force_reprocess": false
}
```

响应：

```json
{
  "status": "success",
  "message": "索引构建成功",
  "files_processed": 10
}
```

### GET /stats

获取系统统计信息：

```json
{
  "processed_files_count": 10,
  "cache_enabled": true,
  "cache_directory": ".cache",
  "config": {
    "adapter": {
      "type": "text"
    },
    // ...其他配置
  }
}
``` 