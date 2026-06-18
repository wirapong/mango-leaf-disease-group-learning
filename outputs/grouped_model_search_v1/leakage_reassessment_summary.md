# Leakage Reassessment Summary

## Why this reassessment was needed
The original 70/15/15 stratified split produced extremely high results, including a 100% soft-voting ensemble. A post-hoc audit of image hashes found duplicate or near-duplicate overlap across train/validation/test. To reduce optimistic bias, a new group-aware split was created.

## Group-aware split construction
- Dataset: 4,000 images across 8 classes
- Split target: ~70/15/15 by class
- Grouping signals used within each class:
  - cleaned filename stem
  - exact average hash (`aHash`)
  - exact difference hash (`dHash`)
- Output split directory: `outputs/grouped_split_v1/`

## Group-aware split audit
- Train images: 2,800
- Validation images: 601
- Test images: 599
- Cross-split overlap after regrouping:
  - `group_id`: 0
  - `clean_stem`: 0
  - `aHash`: 0
  - `dHash`: 0

## Key model results on the stricter split
| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| EfficientNetB0 | 0.994992 | 0.995036 | 0.994992 | 0.995003 |
| MobileNetV3Large | 0.983306 | 0.983616 | 0.983306 | 0.983266 |
| 2-model soft voting (EfficientNetB0 + MobileNetV3Large) | 0.994992 | 0.995100 | 0.994992 | 0.995002 |

## Comparison against the original split
| Model | Original Accuracy | Group-aware Accuracy | Delta |
|---|---:|---:|---:|
| EfficientNetB0 | 0.998333 | 0.994992 | -0.003342 |
| MobileNetV3Large | 0.996667 | 0.983306 | -0.013361 |

## Interpretation
1. The original evaluation was indeed optimistic because leakage existed.
2. After removing direct grouped overlap, EfficientNetB0 still remained very strong.
3. MobileNetV3Large dropped more noticeably, suggesting it benefited more from the original easier split.
4. The stricter split does **not** support the earlier claim of a perfect ensemble result.
5. The project should report the group-aware split as the more credible internal evaluation.

## Remaining caution
This reassessment removes direct grouped overlap detected by filename and exact perceptual hashes, but it is still not the same as external validation on new orchards, new devices, or new acquisition conditions.
