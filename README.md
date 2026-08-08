# NMA Unified Intent Router 交付包

本目录包含 NMA Query Intent Router v1.16.0 四分类离线模型，以及使用
`gpt-5.6-sol` 检测 `service_submission` 的五类统一路由层。

- `POST /predict`：保持原有四分类，完全离线。
- `POST /route`：先检测 submission，非 submission 再调用原模型，对外输出五类之一。

## 文件导航

- `MODEL_CARD.md`：用途、四类标签、模型结构、训练指标和限制。
- `DEPLOYMENT.md`：Docker 部署、API 调用、Chatbot 集成和排障。
- `model/intent_classifier/`：E5 Base 编码器、tokenizer、分类头及元数据。
- `runtime/app.py`：独立 FastAPI 推理服务。
- `runtime/submission_router.py`：GPT submission 检测和五类统一路由。
- `runtime/skills/detect-service-submission.md`：submission 判断 Skill。
- `tests/data/submission_detection.jsonl`：submission/非 submission 二分类评测集。
- `runtime/llm_config.py`：共享 LLM 环境变量和 SDK 客户端配置。
- `scripts/evaluate_submission_detection.py`：真实 submission 二分类评测脚本。
- `scripts/run_service_routing_e2e.py`：连接 `depose` 和 Python Service Resolver 的端到端测试。
- `Dockerfile` / `docker-compose.yml`：容器构建与启动配置。
- `MANIFEST.json`：交付版本和关键指标。
- `SHA256SUMS`：交付文件完整性校验。

## 最短启动路径

```bash
docker compose build
docker compose up -d
curl http://localhost:8000/health
```

GPT 配置位于 `.env`，字段模板见 [`.env.example`](.env.example)。填写
`OPENAI_API_KEY` 后，Docker Compose 会自动加载该文件。`.env` 已加入
`.gitignore`，Key 不会被提交到代码库；生产环境仍建议使用 Secret 管理系统。

```dotenv
OPENAI_BASE_URL=http://gateway.example:8080
OPENAI_API_KEY=
CLAW_LLM_MODEL=gpt-5.6-sol
OPENAI_WIRE_API=responses
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=2
LLM_EVAL_CONCURRENCY=1
```

统一五类结果：

```text
knowledge_process
data_single
data_multi
data_analytics
service_submission
```

`service_submission` 的 `score` 为 `null`，因为 LLM 结构化输出不是经过校准的
分类概率。Flowise 应根据 `intent` 和 `route_source` 路由，不要给该结果设置虚假概率。

内部兼容网关支持 Responses JSON Schema，但不支持 Responses function tool calling，
因此两个 LLM 路由阶段都使用严格 JSON Schema 输出，并由本地代码再次校验目录和状态。

## Submission 评测

补入 `.env` 中的 Key 后运行：

```bash
python3 scripts/evaluate_submission_detection.py
```

评测只检查 submission 与非 submission，不评价原模型内部四类之间的分类。
默认验收门槛为 Precision >= 95% 且 Recall >= 95%。

当前内部网关实测结果：72/72，TP 36、TN 36、FP 0、FN 0，Accuracy、
Precision、Recall 和 F1 均为 100%。三语端到端 smoke 数据为 15/15。

端到端测试要求同级目录存在 `service-router-test`：

```bash
python3 scripts/run_service_routing_e2e.py
```

原四分类模型的正式验收指标仍为固定 Strict Test v3 上的 Macro F1 **89.63%**、
Accuracy **90.00%**。新增 submission 评测与该指标相互独立；详细边界请先阅读
`MODEL_CARD.md`。
