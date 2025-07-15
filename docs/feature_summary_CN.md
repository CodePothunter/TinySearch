# TinySearch 功能概述

本文档提供了TinySearch最近实现的功能概述。

## 数据验证工具

TinySearch现在在`tinysearch.validation`模块中包含了全面的数据验证工具。这些工具有助于确保整个处理流程中的数据完整性：

- **文件和目录验证**：验证文件和目录是否存在并具有正确的格式。
- **嵌入向量验证**：检查嵌入向量是否具有一致的维度和正确的数值。
- **配置验证**：根据必需的键和模式验证配置字典。
- **文本和列表验证**：确保文本字符串和列表不为空。
- **自定义验证**：支持带有清晰错误消息的自定义验证函数。

使用示例：

```python
from tinysearch.validation import DataValidator, ValidationError

# 验证文件是否存在
try:
    file_path = DataValidator.validate_file_exists("data/documents.txt")
    print(f"有效文件: {file_path}")
except ValidationError as e:
    print(f"验证错误: {e}")

# 验证嵌入向量
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
try:
    DataValidator.validate_embeddings(embeddings, expected_dim=3)
    print("有效的嵌入向量")
except ValidationError as e:
    print(f"嵌入向量验证错误: {e}")
```

## 上下文窗口管理

TinySearch现在通过`tinysearch.context_window`模块支持上下文窗口管理。此功能有助于优化LLM处理的内容：

- **Token计数**：估算文本块的token数量。
- **窗口大小调整**：将文本块调整为符合token限制的上下文窗口。
- **窗口合并**：使用可配置的重叠策略合并多个上下文窗口。
- **查询特定上下文**：为特定查询生成最佳上下文窗口。

使用示例：

```python
from tinysearch.context_window import ContextWindowManager

# 初始化上下文窗口管理器
manager = ContextWindowManager(
    max_tokens=4096,
    reserved_tokens=1000,  # 用于提示和响应
    overlap_strategy="smart"
)

# 将文本块调整为窗口大小
text_chunks = ["较长的文本块...", "另一个文本块..."]
metadata_list = [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
windows = manager.fit_text_to_window(text_chunks, metadata_list)

# 为特定查询生成上下文
query = "什么是向量搜索？"
context_text, context_metadata = manager.generate_context_for_query(
    query, text_chunks, metadata_list
)
```

## 响应格式化工具

TinySearch现在在`tinysearch.formatters`模块中包含响应格式化工具，为搜索结果提供多种输出格式：

- **纯文本**：带有可配置分隔符的简单文本格式。
- **Markdown**：带有标题、代码块和元数据部分的富文本格式。
- **JSON**：结构化数据格式，可选择美化打印和时间戳。
- **HTML**：带有内置CSS样式的网页就绪格式。

使用示例：

```python
from tinysearch.formatters import get_formatter

# 从查询引擎获取搜索结果
results = query_engine.retrieve("量子计算", top_k=3)

# 格式化为纯文本
text_formatter = get_formatter("text", include_scores=True)
text_output = text_formatter.format_response(results)
print(text_output)

# 格式化为Markdown
md_formatter = get_formatter("markdown", link_sources=True)
md_output = md_formatter.format_response(results)
print(md_output)

# 格式化为JSON
json_formatter = get_formatter("json", pretty=True)
json_output = json_formatter.format_response(results)
print(json_output)
```

## 热更新功能

TinySearch现在在`tinysearch.flow`模块中集成了热更新功能，当源文档更改时可以实时更新索引：

- **文件变更检测**：自动检测文件的创建、修改、删除或移动。
- **延迟处理**：将更改分组以避免冗余处理。
- **选择性监视**：监视特定的文件扩展名或目录。
- **递归监视**：监视子目录的更改。
- **更新回调**：注册自定义操作的回调函数。

使用示例：

```python
from tinysearch.flow import FlowController

# 初始化组件和流控制器
flow_controller = FlowController(
    data_adapter=data_adapter,
    text_splitter=text_splitter,
    embedder=embedder,
    indexer=indexer,
    query_engine=query_engine,
    config=config_dict
)

# 启动热更新监控
flow_controller.start_hot_update(
    watch_paths=["data/documents"],
    file_extensions=[".txt", ".md", ".pdf"],
    process_delay=1.0,
    recursive=True,
    on_update_callback=lambda updates, deletions: print(f"已更新: {len(updates)}, 已删除: {len(deletions)}")
)

# 需要时，停止热更新监控
flow_controller.stop_hot_update()
```

## 下一步计划

随着这些功能的实现，TinySearch现在是一个更加健壮和灵活的向量检索系统。未来的开发将集中在：

1. Docker部署以简化安装
2. 常见用例的示例配置
3. 性能监控和优化
4. 高级重排序以改进搜索结果
5. 结合向量和基于关键词的混合搜索功能 