# Consistency check: ext_main_20261011-162418 vs base main_20261010-003501

**all_consistent = True**

| comparison | n new | n base | unmatched | mismatching columns |
|---|---|---|---|---|
| CLEAN_clean_vs_CLEAN | 48 | 48 | 0 | none |
| BB_none_vs_A | 384 | 384 | 0 | none |
| RR_unsharp_vs_C | 384 | 384 | 0 | none |
| RR_wiener_vs_D | 384 | 384 | 0 | none |

Separate detect()+compute() vs detectAndCompute (same image): keypoints equal 1584/1584, descriptors equal 1584/1584.

- BB/ORB: 192/192 kp, 192/192 desc
- BB/SIFT: 192/192 kp, 192/192 desc
- CLEAN/ORB: 24/24 kp, 24/24 desc
- CLEAN/SIFT: 24/24 kp, 24/24 desc
- RR/ORB: 576/576 kp, 576/576 desc
- RR/SIFT: 576/576 kp, 576/576 desc

compute() bookkeeping over all stage rows: rows = 3840, rows_with_removed_keypoints = 0, total_removed = 0, rows_reordered = 0, rows_attrs_changed = 0, sift_rows_sentinel_effective = 312, sift_rows = 1920
