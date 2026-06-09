# Code Quality Refactor Plan

## 1. Objective

This document describes the planned code quality improvements for the current fashion fine-grained vision project.

The current prototype has successfully validated the core visual pipeline and the rule-based query-to-region demo. However, several demo scripts still contain mixed responsibilities and should be gradually refactored into reusable modules.

This plan follows the code quality requirements:

1. Modular design.
2. Configuration files instead of hard-coded parameters.
3. Robust exception handling.
4. Clear comments and docstrings.
5. Type hints.
6. Code reuse.
7. PEP8-style naming and formatting.

---

## 2. Current Status

The current project includes several working components:

| Component | Status |
|---|---|
| DeepFashion2 annotation parsing | Working |
| YOLO garment detection | Working |
| SAM-HQ garment segmentation | Working |
| Landmark prediction | Working |
| Semantic region crop generation | Working |
| Mask-aware crop generation | Working |
| Chinese query-to-region demo | Working |
| Batch60 validation | Completed |
| FashionAI attribute baseline | Pending |
| YOLO-seg baseline | Pending |

The main code quality issue is that some demo scripts combine too many responsibilities in a single file.

Example:

```text
tools/demo/query_region_online_demo.py
```

This script currently includes:

- argument parsing
- query alias definitions
- query parsing
- candidate selection
- mask processing
- overlay visualization
- pipeline invocation
- result JSON generation

This is acceptable for quick prototyping, but should be refactored before final delivery.

---

## 3. Refactor Target for Query Region Demo

### 3.1 Current File

```text
tools/demo/query_region_online_demo.py
```

### 3.2 Proposed Modular Structure

```text
tools/
├── demo/
│   └── query_region_online_demo.py
└── query_region/
    ├── __init__.py
    ├── aliases.py
    ├── parser.py
    ├── selection.py
    ├── mask_utils.py
    ├── visualization.py
    └── io_utils.py
```

---

## 4. Module Responsibilities

### 4.1 `aliases.py`

Responsible for query aliases and garment class groups.

Contents:

```text
REGION_ALIASES
SPECIAL_QUERY_ALIASES
COMPONENT_ALIASES
REGION_DISPLAY_NAME
UPPER_BODY_CLASSES
DRESS_CLASSES
LOWER_BODY_CLASSES
```

Purpose:

- Avoid hard-coded alias dictionaries inside the demo script.
- Make it easier to add new query aliases.

---

### 4.2 `parser.py`

Responsible for natural language query parsing.

Main function:

```python
def infer_region_from_query(query: str) -> tuple[str | None, str | None, str, dict[str, Any]]:
    ...
```

Purpose:

- Convert Chinese query into target region.
- Infer component such as `left_sleeve` or `right_sleeve`.
- Infer class constraint for special queries such as `裙摆`.

---

### 4.3 `selection.py`

Responsible for selecting the best candidate from `region_masked_crops.json`.

Main functions:

```python
def garment_group(class_name: Any) -> str:
    ...

def waist_class_priority(class_name: Any) -> int:
    ...

def select_best_record(...) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    ...
```

Purpose:

- Keep all candidate filtering and deterministic selection logic in one reusable module.
- Ensure waist-specific priority only applies to `waist`.
- Keep generic hem and skirt hem separated.

---

### 4.4 `mask_utils.py`

Responsible for mask processing.

Main functions:

```python
def ensure_2d_mask(mask: np.ndarray | None) -> np.ndarray | None:
    ...

def ensure_binary_mask_2d(mask: np.ndarray | None) -> np.ndarray | None:
    ...

def create_full_size_region_mask(...) -> np.ndarray:
    ...
```

Purpose:

- Centralize binary mask conversion.
- Avoid duplicate mask handling code.

---

### 4.5 `visualization.py`

Responsible for visualization output.

Main function:

```python
def draw_selected_overlay(...) -> np.ndarray:
    ...
```

Purpose:

- Draw selected local region overlay.
- Draw bounding box and label.

---

### 4.6 `io_utils.py`

Responsible for common file IO utilities.

Main functions:

```python
def load_json(path: str | Path) -> dict[str, Any]:
    ...

def save_json(data: dict[str, Any], path: str | Path) -> None:
    ...

def copy_if_exists(src: str | None, dst: Path) -> str | None:
    ...

def sanitize_filename_part(text: Any) -> str:
    ...
```

Purpose:

- Avoid repeated file IO logic.
- Improve readability of demo scripts.

---

## 5. Configuration File Plan

The current demo contains hard-coded aliases and default paths.

To improve maintainability, the following configuration files can be added later:

```text
configs/
├── region_query_aliases.yaml
├── query_region_demo.yaml
├── attribute_taxonomy.yaml
└── deepfashion2_category_mapping.yaml
```

### 5.1 `configs/region_query_aliases.yaml`

Example:

```yaml
regions:
  collar:
    - collar
    - neckline
    - 领口
    - 衣领
  sleeve:
    - sleeve
    - 袖子
    - 左袖
    - 右袖
  hem:
    - hem
    - 下摆
    - 衣摆
  waist:
    - waist
    - 腰
    - 腰部
  pant_leg:
    - pant_leg
    - 裤腿
    - 裤管

special_queries:
  skirt_hem:
    - 裙摆
    - 裙子下摆
  dress_hem:
    - 连衣裙下摆
    - 连衣裙裙摆
```

---

## 6. Exception Handling Plan

Current scripts already raise exceptions for critical errors such as missing image or missing JSON file.

Additional improvements:

1. Use clearer error messages when no matching region is found.
2. Save failed query metadata to `result.json`.
3. Avoid stopping batch scripts when one query fails.
4. Add optional `--allow-failed-query` flag for large batch runs.
5. Separate expected semantic failures from unexpected runtime failures.

Expected semantic failure:

```text
Query = 领口
Image = pants only
Reason = no collar region exists
```

Unexpected runtime failure:

```text
Missing mask file
Invalid bbox
Corrupted image
JSON parse error
```

---

## 7. Type Hint and Docstring Plan

All reusable functions should include:

- type hints
- docstrings
- clear return value descriptions

Example:

```python
def target_class_matches(class_name: Any, target_class: str) -> bool:
    """
    Check whether a garment class matches the target class constraint.

    Args:
        class_name: Class name from a candidate crop record.
        target_class: User-provided or query-inferred class constraint.

    Returns:
        True if the candidate class is acceptable, otherwise False.
    """
```

---

## 8. PEP8 and Naming Plan

Recommended conventions:

| Item | Convention |
|---|---|
| Function names | lower_case_with_underscores |
| Variable names | lower_case_with_underscores |
| Constants | UPPER_CASE |
| Class names | PascalCase |
| Config file names | lower_case_with_underscores |
| Output directory names | lower_case_with_underscores |

The current code mostly follows this style, but some long functions should be split.

---

## 9. Short-term Refactor Tasks

Priority order:

| Priority | Task | Status |
|---:|---|---|
| P0 | Keep current demo stable | Done |
| P1 | Add batch summarization script | Pending |
| P2 | Move query aliases to a separate module or config | Pending |
| P3 | Move candidate selection logic to `selection.py` | Pending |
| P4 | Move mask and visualization utilities to modules | Pending |
| P5 | Add unit tests for query parsing and selection | Pending |

---

## 10. Suggested Unit Tests

Future tests can be placed under:

```text
tests/
└── test_query_region.py
```

Test cases:

| Query | Expected Region | Expected Component | Expected Class |
|---|---|---|---|
| 领口 | collar | None | None |
| 左袖子 | sleeve | left_sleeve | None |
| 右袖子 | sleeve | right_sleeve | None |
| 下摆 | hem | None | None |
| 裙摆 | hem | None | skirt |
| 连衣裙下摆 | hem | None | __dress__ |
| 腰部 | waist | None | None |
| 裤腿 | pant_leg | None | None |

Selection tests:

1. Waist should prefer upper garment over lower garment.
2. Skirt hem should only select skirt hem.
3. Generic hem should not force skirt.
4. Left sleeve query should not select right sleeve.
5. Right sleeve query should not select left sleeve.

---

## 11. Long-term Engineering Plan

After the FashionAI attribute baseline is implemented, the codebase should be organized around reusable modules:

```text
tools/
├── infer/
├── data/
├── train/
├── eval/
├── demo/
├── query_region/
└── attributes/
```

Suggested new modules:

```text
tools/attributes/
├── __init__.py
├── dataset.py
├── models.py
├── transforms.py
├── taxonomy.py
└── inference.py
```

This will support the next module:

```text
query region crop
    ↓
attribute classifier
    ↓
structured attribute output
```

---

## 12. Conclusion

The current codebase is acceptable for prototype validation. The next step is not a full rewrite, but a gradual refactor.

Recommended immediate actions:

1. Keep `query_region_online_demo.py` stable.
2. Add a batch result summarization script.
3. Start FashionAI attribute baseline.
4. Refactor query parsing and selection logic after the attribute baseline is working.

The project should prioritize deliverable functionality first, then gradually improve modularity and maintainability.
