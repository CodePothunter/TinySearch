# TinySearch 用户指南

本指南为 TinySearch 提供全面的使用说明，TinySearch 是一个轻量级向量检索系统，专为文本数据的嵌入、索引和搜索而设计。

## 安装

### 基本安装

```bash
pip install tinysearch
```

### 安装附加功能

```bash
# 安装 API 支持
pip install tinysearch[api]

# 安装嵌入模型支持
pip install tinysearch[embedders]

# 安装所有文档适配器
pip install tinysearch[adapters]

# 安装所有功能
pip install tinysearch[full]
```

## 核心概念

TinySearch 由几个关键组件组成，这些组件共同协作以实现高效的文本搜索：

1. **DataAdapter（数据适配器）**：从各种文件格式（TXT、PDF、CSV、Markdown、JSON）中提取文本
2. **TextSplitter（文本分割器）**：将文本分割成适合嵌入的片段
3. **Embedder（嵌入器）**：为文本片段生成向量嵌入
4. **VectorIndexer（向量索引器）**：构建和维护 FAISS 索引以进行高效的相似性搜索
5. **QueryEngine（查询引擎）**：处理查询并检索相关上下文
6. **FlowController（流程控制器）**：编排整个数据流

## 配置

TinySearch 使用 YAML 配置文件控制系统的所有方面。以下是示例配置：

```yaml
# 数据适配器配置
adapter:
  type: text  # 选项：text, pdf, csv, markdown, json, custom
  params:
    encoding: utf-8

# 文本分割器配置
splitter:
  type: character
  chunk_size: 300  # 每个块的字符数
  chunk_overlap: 50  # 块之间的重叠部分
  separator: "\n\n"  # 可选的段落分隔符

# 嵌入模型配置
embedder:
  type: huggingface
  model: Qwen/Qwen-Embedding  # 或任何 HuggingFace 模型
  device: cuda  # 如果没有 GPU，设置为 "cpu"
  normalize: true

# 向量索引器配置
indexer:
  type: faiss
  index_path: index.faiss
  metric: cosine  # 选项：cosine, l2, ip (内积)
  index_type: Flat  # 选项：Flat, IVF, HNSW

# 查询引擎配置
query_engine:
  method: template
  template: "请帮我查找：{query}"
  top_k: 5

# 流程控制器配置
flow:
  use_cache: true
  cache_dir: .cache
```

## 命令行使用

### 索引文档

要从您的文档构建搜索索引：

```bash
tinysearch index --data ./your_documents --config config.yaml
```

选项：
- `--data`：包含文档的文件或目录的路径
- `--config`：配置文件的路径
- `--force`：强制重新处理所有文件，忽略缓存

### 查询

要搜索已索引的文档：

```bash
tinysearch query --q "您的搜索查询" --config config.yaml --top-k 5
```

选项：
- `--q` 或 `--query`：您的搜索查询
- `--config`：配置文件的路径
- `--top-k`：返回结果的数量（覆盖配置文件中的设置）

### API 服务器

要启动 API 服务器：

```bash
tinysearch-api --config config.yaml --port 8000
```

选项：
- `--config`：配置文件的路径
- `--port`：运行服务器的端口（默认：8000）
- `--host`：绑定的主机（默认：127.0.0.1）

## 使用 API

一旦 API 服务器运行，您可以使用 HTTP 请求进行查询：

### 查询端点

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "您的搜索查询", "top_k": 5}'
```

响应格式：

```json
{
  "results": [
    {
      "text": "相关的文本块",
      "score": 0.95,
      "metadata": {
        "source": "/path/to/original/file.txt"
      }
    },
    ...
  ]
}
```

### 索引构建端点

```bash
curl -X POST http://localhost:8000/build-index \
  -H "Content-Type: application/json" \
  -d '{"data_path": "./your_documents", "force_reprocess": false}'
```

## 高级用法

### 自定义数据适配器

您可以通过实现 `DataAdapter` 接口创建自定义数据适配器：

```python
from tinysearch.base import DataAdapter

class MyCustomAdapter(DataAdapter):
    def __init__(self, special_param=None):
        self.special_param = special_param
        
    def extract(self, filepath):
        # 您提取文件文本的代码
        # ...
        return [text1, text2, ...]
```

然后在您的 `config.yaml` 中配置：

```yaml
adapter:
  type: custom
  params:
    module: my_module
    class: MyCustomAdapter
    init:
      special_param: value
```

### 以编程方式使用 FlowController

您可以直接在 Python 代码中使用 TinySearch：

```python
from tinysearch.adapters.text import TextAdapter
from tinysearch.splitters.character import CharacterTextSplitter
from tinysearch.embedders.huggingface import HuggingFaceEmbedder
from tinysearch.indexers.faiss_indexer import FAISSIndexer
from tinysearch.query.template import TemplateQueryEngine
from tinysearch.flow.controller import FlowController

# 创建组件
adapter = TextAdapter()
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedder = HuggingFaceEmbedder(model_name="Qwen/Qwen-Embedding", device="cpu")
indexer = FAISSIndexer()
query_engine = TemplateQueryEngine(indexer=indexer, embedder=embedder)

# 配置
config = {
    "flow": {
        "use_cache": True,
        "cache_dir": ".cache"
    },
    "query_engine": {
        "top_k": 5
    }
}

# 创建 FlowController
controller = FlowController(
    data_adapter=adapter,
    text_splitter=splitter,
    embedder=embedder,
    indexer=indexer,
    query_engine=query_engine,
    config=config
)

# 构建索引
controller.build_index("./your_documents")

# 查询
results = controller.query("您的搜索查询")

# 处理结果
for result in results:
    print(f"分数: {result['score']:.4f}")
    print(f"文本: {result['chunk'].text}")
    print(f"来源: {result['chunk'].metadata.get('source', '未知')}")
    print("---")
```

## Web 界面

TinySearch 包含一个简单的基于 Web 的用户界面，使您无需使用命令行即可轻松搜索已索引的文档并管理索引。

### 启动 Web 界面

Web 界面内置于 API 服务器中。要启动它，请运行：

```bash
tinysearch-api
```

默认情况下，服务器将在 `http://localhost:8000` 上启动。在您的网络浏览器中打开此 URL 以访问用户界面。

### 使用 Web 界面

Web 界面由三个主要部分组成：

#### 搜索

搜索选项卡允许您查询索引。只需输入搜索查询并选择您想看到的结果数量。结果将显示：

- 每个匹配块的文本内容
- 源文档
- 相关性分数（越高越好）

#### 索引管理

索引管理选项卡提供以下工具：

- **上传文档**：上传单个文件以进行索引
- **构建索引**：处理文件目录以构建或更新索引
- **清除索引**：删除所有已索引的文档并重新开始

#### 统计信息

统计选项卡显示有关当前索引的信息，包括：

- 已处理文件的数量
- 是否启用缓存
- 所有已处理文件的列表

### 自定义 Web 界面

Web 界面使用 Bootstrap 5 和原生 JavaScript 构建。如果您想自定义其外观或行为，可以修改 `tinysearch/api/static` 目录中的文件。

## 故障排除

### 常见问题

1. **模型下载失败**：
   - 首次使用 HuggingFace 模型时确保有互联网连接
   - 在嵌入器配置中将 `cache_dir` 设置为可写目录

2. **内存不足错误**：
   - 降低嵌入器配置中的 `batch_size`
   - 使用更小的嵌入模型
   - 处理更小的数据集或减少块大小

3. **索引速度慢**：
   - 在流配置中启用缓存 `use_cache: true`
   - 使用更快的 FAISS 索引类型（如 IVF），但请注意这可能会降低准确性

4. **搜索结果不佳**：
   - 调整块大小和重叠以更好地匹配您的内容
   - 为您的领域使用更适合的嵌入模型
   - 尝试不同的相似度指标（cosine、L2、inner product）

## 支持

如果您遇到任何问题或有疑问，请：

1. 查看[文档](https://github.com/yourusername/tinysearch/docs)
2. 在 [GitHub](https://github.com/yourusername/tinysearch/issues) 上提交问题

## 许可证

TinySearch 基于 MIT 许可证授权。 