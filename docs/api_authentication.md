# TinySearch API Authentication and Rate Limiting Guide

TinySearch API provides secure authentication mechanisms and rate limiting features to protect API access and prevent excessive usage. This document explains how to configure and use these features.

## Table of Contents

1. [Configuring Authentication and Rate Limiting](#configuring-authentication-and-rate-limiting)
2. [API Key Management](#api-key-management)
3. [Using API Keys in Requests](#using-api-keys-in-requests)
4. [Rate Limiting Mechanism](#rate-limiting-mechanism)
5. [Authentication in the Web UI](#authentication-in-the-web-ui)
6. [Example Script](#example-script)

## Configuring Authentication and Rate Limiting

Set authentication and rate limiting parameters in the configuration file:

```yaml
# API configuration
api:
  # Authentication settings
  auth_enabled: true                     # Enable API authentication
  default_key: "your-secure-api-key"     # Default API key
  master_key: "your-master-key"          # Master key for creating new API keys
  
  # Rate limiting settings
  rate_limit_enabled: true               # Enable rate limiting
  rate_limit: 60                         # Maximum number of requests
  rate_limit_window: 60                  # Time window in seconds
```

A complete configuration example can be found in `examples/api_config.yaml`.

## API Key Management

### Generating a New API Key

Create a new API key using the master key:

```bash
curl -X POST "http://localhost:8000/api-key?expires_in_days=30" \
     -H "master-key: your-master-key"
```

Response:

```json
{
  "api_key": "generated-api-key-value",
  "expires_at": "2023-12-31T23:59:59"
}
```

Parameters:
- `expires_in_days`: Days until the API key expires (optional, omit for non-expiring keys)

## Using API Keys in Requests

Add the `X-API-Key` header to all API requests:

```bash
# Query example
curl -X POST "http://localhost:8000/query" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "search content", "top_k": 5}'

# Index building example
curl -X POST "http://localhost:8000/index/build" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"data_path": "/path/to/data", "recursive": true}'
```

## Rate Limiting Mechanism

Rate limiting is implemented using a sliding window algorithm and can be adjusted in the configuration:

- `rate_limit`: Maximum number of requests allowed within the time window
- `rate_limit_window`: Size of the time window in seconds

When rate limits are exceeded, the API returns a `429 Too Many Requests` status code with a `Retry-After` header indicating how long the client should wait.

### Rate Limit Response Example

```
HTTP/1.1 429 Too Many Requests
Retry-After: 5
Content-Type: application/json

{
  "detail": "Rate limit exceeded. Try again in 5 seconds."
}
```

## Authentication in the Web UI

The TinySearch Web UI provides API authentication management features:

1. Click on the "Authentication" tab in the navigation menu
2. Set and save your API key
3. Generate new API keys using the master key

The Web UI automatically includes authentication information in all API requests.

## Example Script

A complete authentication and rate limiting demonstration script is provided in `examples/api_auth_demo.py`, which includes:

1. Starting an API server with authentication and rate limiting
2. Testing different authentication scenarios (no key, invalid key, and valid key)
3. Generating a new API key and testing it
4. Demonstrating rate limiting functionality

Run the example script:

```bash
python examples/api_auth_demo.py
```

## Security Best Practices

1. Always use strong keys and avoid defaults
2. Rotate API keys regularly
3. Enable TLS/HTTPS in production environments
4. Use different API keys for different users for better tracking and revocation
5. Configure reasonable rate limits based on expected usage patterns 