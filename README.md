# NMA Query Intent Router v1.16.0 交付包

本目录可直接交给部署同事，包含完整离线模型和独立 Docker 推理服务。

## 文件导航

- `MODEL_CARD.md`：用途、四类标签、模型结构、训练指标和限制。
- `DEPLOYMENT.md`：Docker 部署、API 调用、Chatbot 集成和排障。
- `model/intent_classifier/`：E5 Base 编码器、tokenizer、分类头及元数据。
- `runtime/app.py`：独立 FastAPI 推理服务。
- `Dockerfile` / `docker-compose.yml`：容器构建与启动配置。
- `MANIFEST.json`：交付版本和关键指标。
- `SHA256SUMS`：交付文件完整性校验。

## 最短启动路径

```bash
docker compose build
docker compose up -d
curl http://localhost:8000/health
```

正式验收指标为固定 Strict Test v3 上的 Macro F1 **89.63%**、Accuracy **90.00%**。当前 E5 编码器未微调，仅训练 MLP 分类头；详细边界请先阅读 `MODEL_CARD.md`。
