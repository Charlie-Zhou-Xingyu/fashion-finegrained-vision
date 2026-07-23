# P0a Vision Adapter Contract

> Version: 1.0.0 | Status: P0a.6.1 | Last updated: 2026-07-14

---

## 1. Current Phase

P0a.6/P0a.6.1 is a **serving adapter skeleton** only.
`MockVisionAttributeProvider` does NOT represent real vision capability.
It exists to define stable input/output contracts so a
`RealVisionAttributeProvider` can be swapped in during P1/P2 without
changing the orchestrator, API schema, or core tests.

---

## 2. Input Semantics

| Field | Type | Meaning |
|---|---|---|
| `image_url` | `str or None` | A URL referencing the garment image. **Not downloaded.** |
| `image_bytes` | `bytes or str or None` | Base64 or raw bytes placeholder. **Not parsed.** Never echoed in response. |
| `regions` | `list[str] or None` | Requested region names, e.g. `["collar", "sleeve", "hem"]`. |
| `attributes` | `dict[str, Any] or None` | Request-provided structured attributes. Higher priority than vision output. |
| `garment_category` | `str or None` | Upstream garment class hint, e.g. `"top"`, `"pants"`. |

---

## 3. Output Semantics

`VisionAttributeResult` fields:

| Field | Meaning | Mock behavior |
|---|---|---|
| `attributes` | Visual attributes extracted by the pipeline. | Always empty `{}`. |
| `garment_instances` | Garment detection results (future). | Always empty `[]`. |
| `regions` | Region localization results (future). | Always empty `[]`. |
| `sources` | Provenance info for extracted data. | Always empty `[]`. |
| `warnings` | Provider-level warnings. | `vision_provider_mock` or `vision_input_missing`. |
| `used_tools` | Tools invoked. | `["mock_vision_provider"]` or `[]`. |
| `meta` | Runtime metadata. | See §5. |

---

## 4. Priority

Fixed, never to be changed:

```
request attributes > visual attributes > unavailable
```

In mock mode, `visual attributes` is always empty, so:

- request attributes → used directly
- no request attributes → `AttributeService` returns unavailable (with `attribute_unavailable` warning)
- no request attributes + `image_url`/`image_bytes` → `vision_provider_mock` warning is also emitted

---

## 5. Meta Fields

Added to `QAOrchestratorResult.meta` (and `MultimodalQAData.meta`):

| Field | Type | Meaning |
|---|---|---|
| `vision_provider_used` | `bool` | Whether the vision provider was invoked. |
| `vision_provider_name` | `str or None` | `"mock"` or `None`. |
| `visual_attributes_present` | `bool` | Whether vision returned non-empty attributes. |
| `provided_attributes_used` | `bool` | Whether request attributes were used directly. |
| `vision_warning_count` | `int` | Number of vision-level warnings. |
| `has_image` | `bool` | Whether raw `image` param was provided. |
| `has_image_url` | `bool` | Whether `image_url` was provided. |
| `has_image_bytes` | `bool` | Whether `image_bytes` was provided. |
| `requested_regions` | `list[str]` | Mirror of the request `regions`. |

---

## 6. Safety Boundaries

- **Do NOT download** `image_url`.
- **Do NOT parse** `image_bytes`.
- **Do NOT return** `image_bytes` content in response/meta/log.
- **Do NOT fabricate** attributes, bboxes, or masks.
- **Do NOT treat** mock output as real vision results.
- `image_bytes` is logged as `bool` only (`has_image_bytes: true/false`).

---

## 7. Warnings Strategy

Warnings only appear at `UnifiedResponse.warnings` (top-level).
They are **NOT** nested inside `data`.

| Warning code | When emitted |
|---|---|
| `vision_provider_mock` | Image source present but real pipeline is not connected. |
| `vision_input_missing` | No image source provided; direct provider call. |
| `attribute_unavailable` | Attribute query received no attributes (from request or vision). |

`vision_input_missing` is NOT emitted for pure knowledge queries or
when the user did not actively provide an image source.

---

## 8. Replacement Strategy

To replace mock with real vision pipeline in P1/P2:

```python
# Before (P0a.6):
provider = MockVisionAttributeProvider()

# After (P2):
provider = RealVisionAttributeProvider(
    yolo_checkpoint="...",
    landmark_checkpoint="...",
)
```

No changes to:
- `QaOrchestrator` routing logic
- `MultimodalQARequest` schema
- `UnifiedResponse` envelope
- Core `tests/test_qa_orchestrator.py` contracts
