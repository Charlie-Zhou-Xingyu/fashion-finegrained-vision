# 服饰多模态问答系统 P0a 阶段架构与模块衔接说明

> **版本:** 1.0.0 | **日期:** 2026-07-14 | **状态:** P0a.5 收口后，准备 P0a.6

---

## 1. 项目当前阶段概览

### 已完成阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0a.1 | FastAPI 基础 + UnifiedResponse / WarningItem / SourceItem schema | ✅ |
| P0a.2 | Rule-Based IntentClassifier (17 类意图) | ✅ |
| P0a.3 | AttributeService (L1 fast-path, 模板回答, fabric sanitise) | ✅ |
| P0a.4 | Knowledge Base seed schema + RagService (exact/alias/BM25) + `/v1/rag/retrieve` | ✅ |
| P0a.5 | QaOrchestrator + `/v1/mm/qa` 真实编排骨架 | ✅ |

### 已实现模块清单

| 模块 | 文件 | 职责 |
|---|---|---|
| FastAPI 服务 | `inference/serving/app.py` | HTTP serving, endpoint registration, request_id, UnifiedResponse |
| API Schema | `inference/serving/schemas.py` | 统一请求/响应模型, WarningItem, SourceItem, disambiguated confidence |
| 意图分类 | `inference/serving/intent_classifier.py` | 17 类意图的规则匹配 (keyword + regex) |
| 属性服务 | `inference/serving/attribute_service.py` | 属性归一化, 模板回答, fabric sanitise, source 路由 |
| RAG 检索 | `inference/serving/rag_service.py` | exact/alias/BM25-like retrieval, category/intent filter |
| QA 编排 | `inference/serving/qa_orchestrator.py` | Intent→Route→Service→Answer 编排, deterministic template |
| 配置 | `configs/*.yaml` | intent taxonomy, attribute templates, KB seed, retrieval config |
| 测试 | `tests/test_serving/` | 220 tests, 全部通过 |

### 当前未实现

- 真实视觉 pipeline 接入
- LLM / MLLM 回答生成
- Redis / FAISS / BGE / reranker
- 完整知识库规模 (1200+ 面料 / 300+ 工艺 / 50+ 风格 / 200+ 术语)
- 正式准确率与延迟评测
- 商家内容生成完整功能
- 多轮对话状态管理

---

## 2. PRD 对齐说明

### 2.1 PRD 3.1 细粒度视觉基础模块

PRD 包含 3.1.1 服饰实例分割, 3.1.2 语言引导局部区域定位, 3.1.3 细粒度属性提取。

**serving 侧当前状态:** 真实视觉模型尚未接入。P0a.6 将通过 `VisionAttributeProvider` adapter 预留接口。

底层算法能力 (`tools/infer/`, `src/fashion_vision/`) 可能已在其他目录实现，但当前 serving 侧未调用。P0a.6 只做 adapter skeleton，不调用真实模型。后续 P1/P2 才接真实 pipeline 和性能指标。

### 2.2 PRD 3.2 多模态大模型推理优化模块

对应 `/v1/mm/qa`, QaOrchestrator, AttributeService, RagService, deterministic answer template。

**已完成:** PRD 3.2.1 多模态问答功能的工程编排骨架。

**未完成:** LLM/MLLM 回答生成, 真实图片理解, 85% QA 准确率评测, 200ms 回答生成正式压测。

### 2.3 PRD 3.2.2 商家内容生成

当前 `content_generation` intent 仅返回 `unsupported`，占位未实现。后续可能进入 P0b ContentGenerationService。

### 2.4 PRD 3.3 多模态 Agent 与 RAG

- IntentClassifier → PRD 3.3.1 工程骨架
- RagService → PRD 3.3.2 工程底座 (15 条 seed KB, exact/alias/BM25-like retrieval)
- QaOrchestrator → 工具调度骨架

**未完成:** 完整 1200+/300+/50+/200+ 知识库, 90% 检索准确率评测, 20ms 检索正式压测, 向量检索/rerank。

---

## 3. 当前代码模块说明

### 3.1 `inference/serving/app.py`

FastAPI 应用入口。注册 6 个端点: `/v1/health`, `/v1/metrics`, `/v1/mm/qa`, `/v1/intent/classify`, `/v1/rag/retrieve`, `/v1/merchant/content/generate`。所有响应包裹为 `UnifiedResponse`，含 `request_id` (支持 `X-Request-ID` 透传)。warnings 仅出现在顶层，不在 `data` 内嵌套。

### 3.2 `inference/serving/schemas.py`

Pydantic v2 模型定义。核心结构:
- `UnifiedResponse` — 统一响应信封
- `WarningItem` — code/scope/message/severity/term/action/reason
- `SourceItem` — type/id/title/category/attribute_confidence/rag_score
- `MultimodalQARequest/Data` — QA 请求/响应
- `RAGRetrieveRequest/Data` — RAG 检索请求/响应

Disambiguated confidence: `intent_confidence`, `attribute_confidence`, `rag_score`, `answer_confidence`, `risk_score` — 不使用单一 `confidence` 字段。

### 3.3 `inference/serving/intent_classifier.py`

`RuleIntentClassifier` 从 `configs/intent_taxonomy.yaml` 加载规则，用 keyword (置信度 0.95) + regex pattern (0.80) 匹配。17 类意图: attribute_query (5), design_explanation (1), craft_explanation (1), styling_advice (2), knowledge_qa (4), content_generation (2), chat (1), fallback_unknown。特定规则 (knowledge_qa, design_explanation) 排在通用规则 (attribute_query, craft_explanation) 之前。未匹配返回 `fallback_unknown` 不报错。

### 3.4 `inference/serving/attribute_service.py`

`AttributeService` 从 `configs/attribute_templates.yaml` 加载模板。输入: attribute_name + raw_attributes dict。归一化三类输入 (AttributeValue / dict / primitive)。Source 路由选择模板 (merchant_input / model_prediction / manual_verified / request_raw)。Fabric sanitise: `composition_verified=False` 去除 `100%`/`百分百` 前缀。Confidence: `answer_confidence = attribute_confidence`, 永不为 1.0。输出 `AttributeAnswer`。

### 3.5 `inference/serving/rag_service.py`

`RagService` 从 `configs/knowledge_base.yaml` (15 条 seed) + `configs/retrieval_config.yaml` 初始化。Strict validation: 缺核心字段/重复 id/source 不在 registry → RuntimeError。检索: query normalize → exact/alias/BM25-like → category filter → dedup/rank → top_k。支持 `attribute_context` 展开 query 为 BM25 提供额外 token。每个 hit 透传 source_ref/review_status/risk_level/allowed_usage。输出 `RetrievalResult`。

### 3.6 `inference/serving/qa_orchestrator.py`

`QaOrchestrator` 接收 query + attributes + garment_category，通过 IntentClassifier 路由。空 query → empty_query answer。attribute_query → AttributeService。knowledge_qa/craft/design → RagService + template。styling_advice → RagService (limited)。content_generation/chat/fallback → unsupported。Warnings 合并, sources 合并, deterministic template。`answer_confidence` 永不为 1.0, `manual_review_required` RAG 知识附带"仍需人工审核"免责。

### 3.7 `inference/serving/deps.py`

Lazy singleton 工厂: `get_qa_orchestrator()`, `get_rag_service()`, `get_attribute_service()`, `get_classifier()`。不在 import 时加载，测试可 monkeypatch。`ServiceState` 暴露 `/v1/health` 的 `implemented_modules` / `pending_modules`。

### 3.8 配置文件

| 文件 | 用途 |
|---|---|
| `configs/serving_config.yaml` | 服务版本, implemented/pending modules, mock 文案 |
| `configs/intent_taxonomy.yaml` | 17 类意图规则 (keyword + regex) |
| `configs/attribute_templates.yaml` | 7 种属性模板, aliases, thresholds |
| `configs/knowledge_base.yaml` | 15 条 seed KB, source registry, schema |
| `configs/retrieval_config.yaml` | BM25/exact/alias 分数, top_k, category boost, intent→category map |
| `configs/knowledge_schema.md` | KB schema 文档 |

---

## 4. 当前核心调用链路

### 4.1 `/v1/mm/qa` 调用链路

```mermaid
flowchart TD
    A[Client] --> B[/v1/mm/qa POST]
    B --> C[FastAPI app]
    C --> D[get_qa_orchestrator]
    D --> E[IntentClassifier.classify]
    E --> F{Intent Route}
    F -->|attribute_query| G[AttributeService.answer_attribute]
    F -->|knowledge_qa/design/craft| H[RagService.retrieve]
    F -->|styling_advice| H
    F -->|content_gen/chat/fallback| I[Template Unsupported]
    G --> J[QAOrchestratorResult]
    H --> K[_knowledge_answer template]
    K --> J
    I --> J
    J --> L[UnifiedResponse]
```

### 4.2 `/v1/rag/retrieve` 调用链路

```mermaid
flowchart TD
    A[Client] --> B[/v1/rag/retrieve POST]
    B --> C[get_rag_service]
    C --> D[query normalize]
    D --> E[exact/alias/title/BM25 search]
    E --> F[category filter + dedup + rank]
    F --> G[RetrievalResult]
    G --> H[UnifiedResponse]
```

### 4.3 Attribute query 示例

"这件衣服是什么面料？" → IntentClassifier → `attribute_query/fabric` → AttributeService.answer_attribute("fabric", {...}) → attribute_answer → UnifiedResponse

### 4.4 Knowledge query 示例

"纤维是什么？" → IntentClassifier → `knowledge_qa/fabric` (pattern `^(纤维).*(是什么)`) → RagService.retrieve → top hit content → deterministic knowledge_answer → UnifiedResponse

---

## 5. Warnings / Sources / Meta 约定

### Warnings 策略

**策略 A: warnings 只在顶层 UnifiedResponse。**

`data` 内不重复 warnings。WarningItem 字段: `code, message, severity, scope`。

常用 warning codes:

- `empty_query` — 空查询
- `unsupported_intent` — 不支持的问题类型
- `low_intent_confidence` — 意图置信度低
- `attribute_unavailable` — 属性缺失
- `no_hits` — RAG 无检索结果
- `top_k_clamped` — top_k 被限制
- `unknown_category` — 未知类别
- `orchestrator_fallback` — 编排器兜底

### Sources

- AttributeService: type="product_attribute", field, value, attribute_confidence, source
- RagService: type="knowledge_base", id, title, category, rag_score, source, source_type, source_ref, review_status

### Meta

- QaOrchestrator: route, primary_intent, sub_intent, attribute_name, rag_hit_count, has_image, has_attributes, garment_category, orchestrator_version
- RagService: effective_categories, top_k, requested_top_k, kb_version, attribute_context_keys, expanded_query

---

## 6. 当前测试体系

| 文件 | 测试数 | 覆盖范围 |
|---|---|---|
| test_app_basic.py | 18 | schema 兼容, HTTP 语义, warnings 位置 |
| test_schemas.py | 25 | Pydantic 模型校验 |
| test_intent_classifier.py | 31 | 17 类意图, 边界, entities |
| test_attribute_service.py | 49 | 归一化, source/confidence/sanitise, 缺失, 未知 |
| test_knowledge_base_schema.py | 23 | KB YAML 解析, 字段, review, source registry |
| test_rag_service.py | 37 | exact/alias/BM25, category, top_k, metadata, singleton |
| test_rag_endpoint.py | 9 | API 集成, no_hits, unknown_category, attribute_context |
| test_qa_orchestrator.py | 18 | empty/attribute/knowledge/styling/unsupported, warnings |
| test_mm_qa_endpoint.py | 10 | QA endpoint, request_id, warnings, JSON, old endpoint regression |
| **合计** | **220** | **all pass** |

---

## 7. 重要设计边界

- 当前不接真实视觉模型
- 当前不接 LLM/MLLM
- 当前不接 Redis/FAISS/BGE
- 当前不调用 `tools/infer/`
- 当前不修改 `src/fashion_vision/`
- 当前知识库是 seed KB, review_status=manual_review_required
- 当前 deterministic answer 不是最终生成式问答
- 当前未达到 PRD 85% QA 准确率
- 当前未达到 PRD 200ms 回答生成正式压测
- 当前未达到完整知识库规模

---

## 8. 后续路线图

| 阶段 | 目标 | 边界 |
|---|---|---|
| P0a.6 | VisionAttributeProvider adapter skeleton | 只做接口骨架, mock provider, 不接真实模型 |
| P0a.7 | Vision context 集成到 QaOrchestrator | 视觉属性作为 QA 上下文, 仍不接真实模型 |
| P0b | ContentGenerationService skeleton | attributes-first 模板方案, 异步 batch, 风险过滤 |
| P1 | RAG Eval Set + KB Expansion | 200+ query-doc 标注集, 500+ seed entries, recall@5/MRR/NDCG |
| P1 | LLM/MLLM Answer Generator Adapter | 独立服务 client, timeout/fallback/circuit_breaker |
| P1 | Latency Benchmark | 生产服务器压测, P50/P95/P99, QPS, GPU 利用率 |
| P2 | Real Vision Pipeline Integration | 连接 tools/infer, 端到端视觉问答 |
| P2 | Vector Retrieval (FAISS/BGE) | Dense retrieval, hybrid search, reranker |

---

## 9. 风险与待决问题

1. **真实视觉 pipeline 接入边界**: 视觉模块接口和 serving 层 schema 需要对齐
2. **attribute schema 与视觉模型输出字段对齐**: 视觉模型输出字段可能和 AttributeValue schema 不匹配
3. **KB 规模与审核流程**: 15 条 seed 远不够；需要领域专家审核 500+ 条目并建立 CI/CD 更新流程
4. **RAG 评测集缺失**: 当前无标注 query-doc pairs，无法正式评估检索准确率
5. **LLM 接入后的事实安全**: LLM 可能编造面料/工艺信息，需要强约束 prompt + 安全过滤
6. **性能指标未压测**: 所有延迟/QPS 指标为估计值，需生产服务器压测
7. **商家内容生成风险**: 夸大宣传/成分比例/品牌名泄漏风险需要充分测试
8. **多轮对话状态管理**: session context、追问检测、话题切换尚未设计

---

## 10. 附录: 当前 API 示例

### `/v1/mm/qa` attribute query

**Request:**
```json
{
  "query": "这件衣服是什么面料？",
  "attributes": {
    "fabric": {
      "value": "棉",
      "attribute_confidence": 0.86,
      "source": "request_raw"
    }
  },
  "garment_category": "shirt"
}
```

**Response:**
```json
{
  "request_id": "req_a1b2c3d4e5f6",
  "status": "success",
  "data": {
    "answer": "这件shirt的请求中标注面料为棉。具体成分建议以商品详情页或水洗标为准。",
    "answer_type": "attribute_answer",
    "answer_confidence": null,
    "intent_confidence": 0.95,
    "sources": [
      {
        "type": "product_attribute",
        "field": "fabric",
        "value": "棉",
        "attribute_confidence": 0.86,
        "source": "request_raw"
      }
    ],
    "is_cached": false,
    "need_image": false
  },
  "elapsed_ms": 2.3,
  "used_tools": ["intent_classifier", "attribute_service"],
  "warnings": [],
  "meta": {
    "path": "attribute_query",
    "schema_version": "1.0.0"
  }
}
```

### `/v1/rag/retrieve` knowledge query

**Request:**
```json
{
  "query": "纤维是什么",
  "top_k": 3,
  "primary_intent": "knowledge_qa",
  "sub_intent": "term"
}
```

**Response:**
```json
{
  "request_id": "req_b2c3d4e5f6a1",
  "status": "success",
  "data": {
    "query": "纤维是什么",
    "normalized_query": "纤维 是什么",
    "hits": [
      {
        "id": "fiber_term_001",
        "category": "fiber",
        "term": "Fiber",
        "zh_term": "纤维",
        "title": "纤维（Fiber）",
        "content_snippet": "纤维是转化为纱线的材料...",
        "score": 1.0,
        "match_type": "exact",
        "source": "materials_terminology_guide_2020",
        "source_type": "pdf",
        "source_ref": {
          "document_title": "Materials Terminology Guide 2020",
          "page_start": 5,
          "page_end": 5,
          "section": "Fiber & Materials Terminology / Material"
        },
        "review_status": "manual_review_required",
        "reviewed_by": null,
        "last_reviewed_at": null,
        "risk_level": "low"
      }
    ],
    "meta": {
      "effective_categories": [],
      "top_k": 3,
      "requested_top_k": 3,
      "kb_version": "1.0.0"
    }
  },
  "elapsed_ms": 1.5,
  "used_tools": ["rag_service"],
  "warnings": [],
  "meta": {
    "path": "rag",
    "schema_version": "1.0.0"
  }
}
```

---

*End of document.*
