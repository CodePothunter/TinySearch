# TinySearch 配置指南

TinySearch 使用灵活的基于 YAML 的配置系统，允许您自定义系统的所有方面。本指南解释了可用的配置选项及其有效使用方法。

## 配置文件结构

TinySearch 配置文件按照每个主要组件划分为不同部分：

```yaml
# 数据适配器配置
adapter:
  type: text
  params:
    # 适配器特定参数

# 文本分割器配置
splitter:
  type: character
  # 分割器特定参数

# 嵌入模型配置
embedder:
  type: huggingface
  # 嵌入器特定参数

# 向量索引器配置
indexer:
  type: faiss
  # 索引器特定参数

# 查询引擎配置
query_engine:
  method: template
  # 查询引擎特定参数

# 流程控制器配置
flow:
  # 流程控制器特定参数
```

## 完整配置示例

这里是一个包含所有可用选项的完整配置文件：

```yaml
# 数据适配器配置
adapter:
  type: text  # 选项：text, pdf, csv, markdown, json, custom
  params:
    encoding: utf-8  # 对于文本文件
    # CSV 特定参数
    # column: "text"
    # delimiter: ","
    # JSON 特定参数
    # key_path: "content.text"
    # collection_key: "items"
    # 自定义适配器参数
    # module: "my_module"
    # class: "MyCustomAdapter"
    # init:
    #   custom_param1: value1
    #   custom_param2: value2

# 文本分割器配置
splitter:
  type: character  # 目前仅支持 "character"
  chunk_size: 300  # 每个块的字符数
  chunk_overlap: 50  # 块之间的重叠部分
  separator: "\n\n"  # 可选的分块分隔符
  keep_separator: false  # 是否在块中保留分隔符
  strip_whitespace: true  # 是否去除块中的空白

# 嵌入模型配置
embedder:
  type: huggingface  # 目前仅支持 "huggingface"
  model: "Qwen/Qwen-Embedding"  # HuggingFace 模型名称
  device: "cuda"  # "cuda" 或 "cpu"
  max_length: 512  # 最大序列长度
  batch_size: 8  # 用于嵌入生成的批处理大小
  normalize: true  # 是否归一化嵌入向量
  cache_dir: "~/.cache/tinysearch/models"  # 模型缓存目录

# 向量索引器配置
indexer:
  type: faiss  # 目前仅支持 "faiss"
  index_type: "Flat"  # 选项："Flat", "IVF", "HNSW"
  metric: "cosine"  # 选项："cosine", "l2", "ip"（内积）
  index_path: ".cache/index.faiss"  # 保存/加载索引的路径（默认：.cache/index.faiss）
  nlist: 100  # IVF 索引的集群数量
  nprobe: 10  # 搜索时检查的集群数量
  use_gpu: false  # 是否使用 GPU 进行索引

# 查询引擎配置
query_engine:
  method: template  # 目前仅支持 "template"
  template: "请帮我查找：{query}"  # 用于格式化查询的模板
  top_k: 5  # 默认返回结果数量

# 流程控制器配置
flow:
  use_cache: true  # 是否使用缓存
  cache_dir: ".cache"  # 缓存目录
```

## 组件特定配置

### 数据适配器

#### TextAdapter（文本适配器）

```yaml
adapter:
  type: text
  params:
    encoding: utf-8  # 文件编码
```

#### PDFAdapter（PDF 适配器）

```yaml
adapter:
  type: pdf
  params:
    # PDF 适配器没有特定参数
```

#### CSVAdapter（CSV 适配器）

```yaml
adapter:
  type: csv
  params:
    column: "text"  # 包含文本数据的列
    encoding: "utf-8"  # 文件编码
    delimiter: ","  # 列分隔符
```

#### MarkdownAdapter（Markdown 适配器）

```yaml
adapter:
  type: markdown
  params:
    # Markdown 适配器没有特定参数
```

#### JSONAdapter（JSON 适配器）

```yaml
adapter:
  type: json
  params:
    key_path: "content.text"  # JSON 结构中文本字段的路径
    collection_key: "items"  # 可选的项目列表路径
```

#### 自定义适配器

```yaml
adapter:
  type: custom
  params:
    module: "my_module"  # 包含适配器的 Python 模块
    class: "MyCustomAdapter"  # 适配器的类名
    init:  # 传递给适配器 __init__ 的参数
      param1: value1
      param2: value2
```

### 文本分割器

```yaml
splitter:
  type: character
  chunk_size: 300  # 每个块的字符数
  chunk_overlap: 50  # 块之间的重叠部分
  separator: "\n\n"  # 可选分隔符（如段落）
  keep_separator: false  # 是否在块中保留分隔符
  strip_whitespace: true  # 是否去除块中的空白
```

### 嵌入器

```yaml
embedder:
  type: huggingface
  model: "Qwen/Qwen-Embedding"  # HuggingFace 模型名称或路径
  device: "cuda"  # "cuda" 或 "cpu" 或特定设备，如 "cuda:0"
  max_length: 512  # 模型的最大 token 长度
  batch_size: 8  # 用于提高效率的批处理大小
  normalize: true  # 是否将嵌入向量归一化为单位长度
  cache_dir: "~/.cache/tinysearch/models"  # 模型缓存目录
```

### 向量索引器

```yaml
indexer:
  type: faiss
  index_type: "Flat"  # "Flat"（精确搜索）, "IVF"（近似）, "HNSW"（图形）
  metric: "cosine"  # "cosine", "l2"（欧几里得）, "ip"（内积）
  index_path: "index.faiss"  # 保存/加载索引的路径
  nlist: 100  # IVF 索引的集群数量（仅适用于 IVF）
  nprobe: 10  # 搜索的集群数量（仅适用于 IVF）
  use_gpu: false  # 是否使用 GPU 加速
```

### 查询引擎

```yaml
query_engine:
  method: template
  template: "请帮我查找：{query}"  # 带有 {query} 占位符的模板
  top_k: 5  # 默认返回结果的数量
```

### 流程控制器

```yaml
flow:
  use_cache: true  # 是否对已处理文件使用缓存
  cache_dir: ".cache"  # 缓存存储目录
```

## 配置加载

TinySearch 按照以下优先顺序加载配置：

1. **默认配置**：内置默认值
2. **配置文件**：来自您配置文件的设置
3. **运行时覆盖**：命令行参数或 API 参数

## 使用环境变量

您可以使用 `${ENV_VAR}` 语法在配置中使用环境变量：

```yaml
embedder:
  model: "${EMBEDDING_MODEL}"
  device: "${DEVICE:-cpu}"  # 如果 DEVICE 未设置，默认为 "cpu"
```

## 配置管理 API

您也可以以编程方式管理配置：

```python
from tinysearch.config import Config

# 从文件加载配置
config = Config("config.yaml")

# 访问配置值
model_name = config.config["embedder"]["model"]

# 修改配置
config.config["embedder"]["device"] = "cpu"

# 保存修改后的配置
config.save("new_config.yaml")
```

## 最佳实践

### 对于小型文档

```yaml
splitter:
  chunk_size: 200
  chunk_overlap: 20

embedder:
  batch_size: 16
```

### 对于大型文档

```yaml
splitter:
  chunk_size: 500
  chunk_overlap: 100

embedder:
  batch_size: 4
```

### 为了速度

```yaml
indexer:
  index_type: "IVF"
  metric: "ip"  # 内积比余弦更快
  nlist: 100
  nprobe: 5  # 更低以加快搜索

flow:
  use_cache: true
```

### 为了准确性

```yaml
indexer:
  index_type: "Flat"  # 精确搜索
  metric: "cosine"  # 余弦对语义相似性更稳健

splitter:
  chunk_size: 300
  chunk_overlap: 150  # 更高的重叠保留上下文
```

## 故障排除

### 配置验证

TinySearch 在加载时验证您的配置。如果有错误，请检查：

1. **无效的组件类型**：确保您使用的是受支持的组件类型
2. **缺少必需参数**：检查是否提供了所有必需的参数
3. **类型不匹配**：确保参数类型与预期一致

### 常见问题

1. **索引未找到**：确保 `index_path` 指向正确位置
2. **模型下载错误**：检查 `cache_dir` 是否可写且您有互联网访问权限
3. **内存不足**：减小嵌入器配置中的 `batch_size` 或 `max_length`
4. **性能缓慢**：考虑更改 `index_type` 或调整 `chunk_size` 