# Knowledge Base Schema

> Version: 1.0.0 | Status: P0a.4 seed KB

---

## 1. Overview

The knowledge base is stored as a single YAML file (`configs/knowledge_base.yaml`).
It provides a minimal set of structured documents (seed entries) extracted from
authoritative industry sources.  These entries are used by the P0a RagService for:

- Exact-match / alias lookup
- BM25 keyword retrieval

The seed KB is **NOT** a production-verified knowledge base.  All entries require
manual review before production use.

---

## 2. Top-Level Structure

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | string | Yes | Schema version (semver). |
| `locale` | string | Yes | Default locale for content (e.g. `zh-CN`). |
| `source_policy` | dict | Yes | Metadata about sourcing and review requirements. |
| `documents` | list | Yes | List of document entries (see §3). |

### `source_policy`

| Field | Type | Description |
|---|---|---|
| `default_review_status` | string | Must be `"manual_review_required"` for seed entries. |
| `notes` | string | Human-readable notes about the seed KB status. |

---

## 3. Document Entry Fields

Every entry in `documents` MUST contain the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | **Yes** | Globally unique, snake_case, no spaces. |
| `category` | string | **Yes** | One of the allowed categories (see §4). |
| `term` | string | **Yes** | Original English term from the source. |
| `zh_term` | string | **Yes** | Consumer-friendly Chinese translation. |
| `aliases` | list[str] | **Yes** | Synonyms in English and Chinese. |
| `title` | string | **Yes** | Short descriptive title for display. |
| `content` | string | **Yes** | Consumer-friendly explanation in Chinese (1-3 sentences). Must NOT contain absolute claims. |
| `allowed_usage` | list[str] | **Yes** | One or more of: `knowledge_qa`, `explanation`, `internal_test`. |
| `risk_level` | string | **Yes** | `low`, `medium`, or `high`. |
| `source` | string | **Yes** | Canonical source identifier (e.g. `materials_terminology_guide_2020`). |
| `source_type` | string | **Yes** | Format of the source (e.g. `pdf`). |
| `source_url` | string or null | **Yes** | URL if available; `null` otherwise. |
| `source_ref` | dict | **Yes** | Structured reference with `document_title`, `page_start`, `page_end`, `section`. |
| `review_status` | string | **Yes** | Must be `"manual_review_required"` for all seed entries. |
| `reviewed_by` | string or null | **Yes** | `null` until reviewed. |
| `last_reviewed_at` | string or null | **Yes** | ISO 8601 date or `null`. |
| `version` | string | **Yes** | Entry version (semver). |
| `tags` | list[str] | **Yes** | Search / filter tags. |

### `source_ref` sub-fields

| Field | Type | Required | Description |
|---|---|---|---|
| `document_title` | string | **Yes** | Full title of the source document. |
| `page_start` | int or null | **Yes** | Starting page number (`null` if unknown). |
| `page_end` | int or null | **Yes** | Ending page number (`null` if unknown). |
| `section` | string or null | **Yes** | Section heading within the document. |

### `allowed_usage` values

| Value | Meaning |
|---|---|
| `knowledge_qa` | May be used in RAG-augmented answers to user queries. |
| `explanation` | May be used in design / craft explanations. |
| `internal_test` | For engineering testing only; not for production use. |

### `risk_level` values

`risk_level` 表示**该知识条目用于用户回答时的误导、合规、可持续宣称或事实错误风险**，
不是材料本身的物理风险或质量等级。

| Value | Meaning | Recommendation |
|---|---|---|
| `low` | 基础术语解释；不涉及可持续/合规宣称。 | 可直接用于模板回答。 |
| `medium` | 涉及可持续、回收、优选、循环、认证、供应链等容易被误读的内容。 | 回答时需附加来源说明，提醒用户以最新标准为准。 |
| `high` | 涉及强认证、功能性、合规承诺、健康/安全/功效等内容。 | 必须人工审核后方可用于生产回答；P0 阶段仅用于内部测试。 |

---

### `allowed_usage` 与 `review_status` 的关系

`allowed_usage` 表示该条目可以**作为检索候选**进入对应场景（knowledge_qa / explanation / internal_test），
**不表示** `review_status=manual_review_required` 的内容可以被最终回答层直接采信。

最终回答层（QaOrchestrator / ContentService）必须结合以下三个字段决定是否使用检索结果：

1. `review_status` — 是否为人工审核通过的条目
2. `risk_level` — 内容的潜在风险等级
3. `allowed_usage` — 该条目允许用于哪些场景

例如：一条 `review_status=manual_review_required` + `risk_level=medium` 的条目可以出现在检索结果中，但回答层必须附加免责声明（如 "以下信息来源于行业参考资料，尚未经过人工审核，仅供参考"）。

---

## 4. Allowed Categories

For P0a.4:

| Category | Description |
|---|---|
| `fabric` | Fabrics and textile materials. |
| `material` | Raw materials and material types. |
| `fiber` | Specific fiber types. |
| `sustainability` | Sustainability concepts and programs. |
| `supply_chain` | Supply chain terminology. |
| `term` | General fashion / textile terminology. |

---

## 5. Review Status Lifecycle

```
seed_unreviewed → manual_review_required → reviewed → deprecated
```

All P0a.4 seed entries use `manual_review_required`.  They must NOT be presented
to end-users as verified facts without explicit human review.

---

## 6. Content Constraints

Seed entries MUST NOT contain:

- `"保证"` (guarantee)
- `"绝对"` (absolute)
- `"一定"` (certainly)
- `"100%"` (unverified percentage)
- `"完全"` (completely)
- `"最环保"` (most environmentally friendly)
- `"最可持续"` (most sustainable)
- `"官方认证为最佳"` (officially certified as best)

Fabric / material descriptions must remind users that visual identification alone
cannot confirm composition; they should consult the product label or care tag.

---

## 7. Source Registry

The top-level `sources` dict records copyright and provenance information for
every external source referenced by documents.  Each `document.source` key
MUST have a corresponding entry in `sources`.

### Required fields per source entry

| Field | Type | Description |
|---|---|---|
| `title` | string | Full title of the source document. |
| `publisher` | string | Publisher / organisation name. |
| `year` | int | Publication year. |
| `version` | string | Version identifier. |
| `copyright` | string | Copyright notice. |
| `source_type` | string | Format (e.g. `pdf`). |
| `source_url` | string or null | URL if available. |
| `license` | string | License identifier; `"unknown"` is acceptable for seed KB. |
| `usage_note` | string | Human-readable usage constraints. |

### Example

```yaml
sources:
  materials_terminology_guide_2020:
    title: "Materials Terminology Guide 2020"
    publisher: "Textile Exchange"
    year: 2020
    version: "Version 1"
    copyright: "Textile Exchange © 2020"
    source_type: "pdf"
    source_url: null
    license: "unknown"
    usage_note: "Used as an internal reference for manually reviewed seed entries. Do not expose verbatim text without permission review."
```

---

## 8. RagService Responsibility

The upcoming `RagService` (P0a.4.2) will:

- Load documents from `knowledge_base.yaml`
- Provide exact-match, alias, and BM25 lookup
- Return matched documents with full metadata (including `source`, `review_status`, `version`)
- **Never** judge the truthfulness of knowledge content — that is a human review responsibility

---

## 8. Why Seed KB ≠ Production Verified KB

- Seed entries are extracted by an engineer, not a domain expert.
- Translations may not be authoritative.
- Content has not been reviewed by a fashion / textile professional.
- The PDF source (Materials Terminology Guide 2020) may be outdated.
