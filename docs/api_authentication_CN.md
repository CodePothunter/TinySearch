# TinySearch API 认证与速率限制指南

TinySearch API提供了安全的认证机制和速率限制功能，用于保护API访问并防止过度使用。本文档介绍了如何配置和使用这些功能。

## 目录

1. [配置认证和速率限制](#配置认证和速率限制)
2. [API密钥管理](#api密钥管理)
3. [在请求中使用API密钥](#在请求中使用api密钥)
4. [速率限制机制](#速率限制机制)
5. [Web界面中的认证管理](#web界面中的认证管理)
6. [示例脚本](#示例脚本)

## 配置认证和速率限制

在配置文件中设置认证和速率限制参数：

```yaml
# API配置
api:
  # 认证设置
  auth_enabled: true                     # 启用API认证
  default_key: "your-secure-api-key"     # 默认API密钥
  master_key: "your-master-key"          # 创建新API密钥的主密钥
  
  # 速率限制设置
  rate_limit_enabled: true               # 启用速率限制
  rate_limit: 60                         # 最大请求数量
  rate_limit_window: 60                  # 时间窗口(秒)
```

完整配置示例可在 `examples/api_config.yaml` 中找到。

## API密钥管理

### 生成新的API密钥

使用主密钥创建新的API密钥：

```bash
curl -X POST "http://localhost:8000/api-key?expires_in_days=30" \
     -H "master-key: your-master-key"
```

返回结果：

```json
{
  "api_key": "generated-api-key-value",
  "expires_at": "2023-12-31T23:59:59"
}
```

参数说明：
- `expires_in_days`: API密钥过期天数（可选，不提供则永不过期）

## 在请求中使用API密钥

在所有API请求的头部添加`X-API-Key`：

```bash
# 查询示例
curl -X POST "http://localhost:8000/query" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "搜索内容", "top_k": 5}'

# 索引构建示例
curl -X POST "http://localhost:8000/index/build" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"data_path": "/path/to/data", "recursive": true}'
```

## 速率限制机制

速率限制基于滑动窗口算法实现，可在配置中调整：

- `rate_limit`: 在时间窗口内允许的最大请求数
- `rate_limit_window`: 时间窗口大小(秒)

当超过速率限制时，API返回`429 Too Many Requests`状态码，并在响应头中包含`Retry-After`字段，指示客户端等待时间。

### 速率限制响应示例

```
HTTP/1.1 429 Too Many Requests
Retry-After: 5
Content-Type: application/json

{
  "detail": "Rate limit exceeded. Try again in 5 seconds."
}
```

## Web界面中的认证管理

TinySearch Web界面提供了API认证管理功能：

1. 在导航菜单中点击"Authentication"选项卡
2. 可以设置和保存API密钥
3. 使用主密钥生成新的API密钥

Web界面会自动在所有API请求中添加认证信息。

## 示例脚本

在`examples/api_auth_demo.py`中提供了完整的认证和速率限制演示脚本，包括：

1. 启动带有认证和速率限制的API服务器
2. 测试不同认证场景(无密钥、无效密钥和有效密钥)
3. 生成新的API密钥并测试
4. 演示速率限制功能

运行示例脚本：

```bash
python examples/api_auth_demo.py
```

## 安全最佳实践

1. 始终使用强密钥，避免默认值
2. 定期轮换API密钥
3. 在生产环境中启用TLS/HTTPS
4. 对不同用户使用不同的API密钥，便于跟踪和撤销
5. 根据预期使用模式配置合理的速率限制 