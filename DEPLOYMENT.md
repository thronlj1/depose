# NMA 查询意图路由部署手册

## 1. 部署结果

本目录是可独立交付的离线推理包。Docker 镜像构建完成后，运行期间不会从 Hugging Face 下载模型，也不依赖原训练平台或其数据卷。

建议资源：2 CPU、4 GiB 内存、至少 4 GiB 可用磁盘。CPU 可以运行；若使用 NVIDIA GPU，需要改用 CUDA 版 PyTorch 基础环境。默认镜像安装 CPU 版 PyTorch。

## 2. 快速启动

在 `depose` 目录执行：

```bash
docker compose build
docker compose up -d
docker compose ps
```

默认监听 `http://localhost:8000`。修改端口：

```bash
INTENT_PORT=18000 docker compose up -d
```

首次构建需要联网下载 Python/PyTorch 依赖；镜像构建完成后，推理运行可以离线。

## 3. 验证部署

健康检查：

```bash
curl http://localhost:8000/health
```

预测法规/流程查询：

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"许可续期需要准备哪些材料？"}'
```

预测数据查询：

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"查询申请 A123 的状态"}'
```

响应示例：

```json
{
  "intent": "data_single",
  "score": 0.91,
  "predictions": [
    {"intent": "data_single", "score": 0.91},
    {"intent": "data_multi", "score": 0.06}
  ],
  "model_version": "v1.16.0"
}
```

`predictions` 实际会返回全部四类，此处仅为缩短示例。

## 4. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活状态、版本、运行设备 |
| GET | `/metadata` | 标签、基础模型与正式指标 |
| POST | `/predict` | 输入 `{"text":"..."}`，返回 top-1 和四类得分 |
| GET | `/docs` | FastAPI 调试文档；生产环境可在网关关闭 |

输入 `text` 去除首尾空格后不能为空，最大 4,000 字符；模型内部最多处理 128 tokens。

## 5. 与 Chatbot 集成

推荐由编排层完成以下逻辑：

```text
1. guard.check(user_text)；非法则立即拒绝
2. 从已验证 token 获取 role、user_id、department、data_scope
3. POST /predict 获取 intent
4. knowledge_process → RAG
5. data_* → 只读 Tool/API Planner
6. Planner 实际选择 1 个 API → data_single
7. Planner 实际选择多个 API → data_multi
8. 涉及统计/比较/计算 → data_analytics
9. 每个 API 再做对象级权限校验和审计
```

不要把 customer/admin 拼入标签，也不要依赖模型决定权限。当前交付仅允许接入查询接口，不能配置审批、删除、更新状态、任务分配、发布或广播等写操作。

## 6. 配置项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INTENT_PORT` | `8000` | Docker Compose 暴露端口 |
| `INTENT_DEVICE` | `auto` | `auto`、`cpu` 或 `cuda`；默认镜像适合 CPU |
| `TORCH_INDEX_URL` | PyTorch CPU index | 构建时的 PyTorch wheel 地址 |
| `MODEL_DIR` | `/model/intent_classifier` | 容器内模型目录，Dockerfile 已固定 |

生产环境建议保持 Uvicorn 单 worker：每个 worker 都会单独加载约 1.1 GB 权重。横向扩容应增加容器副本，而不是在单容器内盲目增加 worker。

## 7. 生产加固

- 服务放在内部网络，通过 API Gateway 提供 TLS、身份验证、速率限制和请求体限制。
- 不记录 token 和完整敏感问题；日志使用 correlation ID，并对输入做脱敏。
- API 层继续执行 RBAC 和对象级数据范围过滤，不能信任 Chatbot 传来的角色文本。
- 监控各标签分布、低置信度比例、人工改判率、延迟和错误率。
- 将 `data_single`/`data_multi` 的 Planner 改判记录回流为训练样本，但不要把固定测试集回流训练。
- 模型健康检查成功只表示权重已加载，不代表业务下游 API 正常。

## 8. 替换模型

新模型必须保持以下目录契约：

```text
model/intent_classifier/
├── encoder/config.json
├── encoder/model.safetensors
├── tokenizer/*
├── head.pt
└── metadata.json
```

替换后更新 `metadata.json` 的版本和结构参数，重新执行：

```bash
docker compose build --no-cache
docker compose up -d
```

先用固定 Strict Test v3 验收，再做小流量灰度。若采用 E5 局部/全量微调，必须交付微调后的 `encoder/model.safetensors`，不能继续使用原始 E5 Base 权重。

## 9. 故障排查

- `Missing model metadata`：确认 `model/intent_classifier/metadata.json` 存在。
- 启动时被系统杀死：通常是内存不足，提升至至少 4 GiB，且保持单 worker。
- 构建时无法访问 PyTorch 源：通过 `TORCH_INDEX_URL` 指向公司镜像源。
- `INTENT_DEVICE=cuda, but CUDA is unavailable`：当前环境没有可用 CUDA，改为 `cpu`/`auto`，或构建 CUDA 版镜像。
- single/multi 结果不稳定：让 Planner 按实际 API 调用数量最终校正，不要反复调用模型投票。

## 10. 完整性校验

交付接收方在 `depose` 目录运行：

```bash
shasum -a 256 -c SHA256SUMS
```

全部显示 `OK` 后再构建镜像。
