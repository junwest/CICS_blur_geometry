# Consistency check: ext_smoke_main_20261010-223440 vs base main_20261010-003501

**all_consistent = True**

| comparison | n new | n base | unmatched | mismatching columns |
|---|---|---|---|---|
| CLEAN_clean_vs_CLEAN | 2 | 2 | 0 | none |
| BB_none_vs_A | 16 | 16 | 0 | none |
| RR_unsharp_vs_C | 16 | 16 | 0 | none |
| RR_wiener_vs_D | 16 | 16 | 0 | none |

Separate detect()+compute() vs detectAndCompute (same image): keypoints equal 66/66, descriptors equal 66/66.

- BB/ORB: 8/8 kp, 8/8 desc
- BB/SIFT: 8/8 kp, 8/8 desc
- CLEAN/ORB: 1/1 kp, 1/1 desc
- CLEAN/SIFT: 1/1 kp, 1/1 desc
- RR/ORB: 24/24 kp, 24/24 desc
- RR/SIFT: 24/24 kp, 24/24 desc

compute() bookkeeping over all stage rows: rows = 160, rows_with_removed_keypoints = 0, total_removed = 0, rows_reordered = 0, rows_attrs_changed = 0, sift_rows_sentinel_effective = 6, sift_rows = 80
