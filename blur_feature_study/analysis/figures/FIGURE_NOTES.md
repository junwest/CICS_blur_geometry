# 분석용 Figure 설명 (검증·탐색용, 논문 Figure 아님)

모든 그림은 `analysis/scripts/06_figures.py`(통계 그림), `07_visual_examples.py`(영상 예시)로 생성했다. 데이터는 Test split(24쌍, 12 시퀀스)의 `results/main/main_20261010-003501/raw_results.csv`이며, 별도 표기가 없으면 Success는 Success@3px, GT-correct는 3px 기준이다. 색은 고정이다: A 파랑, C 주황, D 청록, B 노랑, Clean 회색 점선. 검출기는 색이 아니라 패널로 구분한다. 신뢰구간(CI)은 모두 시퀀스 단위 클러스터 부트스트랩 95%다.

| 그림 | ① 무엇을 비교 | ② 데이터·조건 | ③ 관찰 | ④ 이 그림만으로 알 수 없는 것 |
|---|---|---|---|---|
| F01_profile_by_blur | 블러 조건별 Success@3(위)·GT-correct(아래), 그룹 CLEAN/A/B/C/D | 24쌍 평균, ORB·SIFT 패널 | D가 모든 조건에서 A보다 높고 ORB에서는 Clean 수준에 근접. C는 Gaussian/Disk에서만 A보다 뚜렷이 높음. B(노랑)는 A와 거의 겹침. 모션 17px에서 A 성공률이 ORB 0.12–0.17, SIFT 0.42–0.62로 급락 | 평균만 보여 쌍 간 분산·대응 관계가 없음. 성공률은 24쌍 이진값 평균이라 ±0.04 단위로 움직임 |
| F02_paired_effects_forest | 같은 셀에서의 차이 B−A, C−A, D−A (Success, GT-correct) | 블러 조건별 24쌍 대응 차이 + CI | D−A는 GT-correct에서 모든 조건의 CI가 0을 크게 벗어남. Success는 모션 17px에서만 CI가 0을 벗어나고(ORB +0.58, SIFT +0.33~0.46) 약한 블러에서는 CI가 0을 포함(천장 효과). B−A, C−A의 Success CI는 거의 모두 0 포함 | 왜 그런지는 설명하지 않음. Success CI는 n=24 이진값이라 넓음 |
| F03_processing_effects_all_metrics | C−A(주황)와 D−A(청록)를 6개 지표에서 동시에 | 블러 조건별 대응 차이 + CI | **Unsharp의 해리:** 모션 블러에서 PSNR은 감소(CI<0)하고 SSIM≈0인데 Edge F1·Repeatability는 모든 조건에서 증가, GT-correct는 모션 17px에서 0. D는 모든 지표가 함께 상승 | 지표 간 인과 방향. Edge F1 상승이 가짜 엣지 때문인지는 F11에서 확인 |
| F04_psnr_ssim_vs_edgef1 | 영상 수준 PSNR/SSIM vs Edge F1 | 처리 영상 576장(A/C/D), 마커 = 블러 계열 | 모션 블러(삼각형)는 PSNR이 낮은데 Edge F1이 높음. Gaussian/Disk 강한 블러는 PSNR 23 dB 근방에서 Edge F1≈0.1. 전체 Spearman ρ는 PSNR 0.31, SSIM 0.53 | 같은 쌍의 여러 점이 독립이 아님(쌍당 24점). 상관값은 서술용 |
| F05_metrics_vs_log_gt_ratio | 4개 지표 vs clean 대비 GT-correct 비율(log) | 검출기별 576행(A/C/D) | Repeatability가 가장 조밀(ρ ORB 0.76, SIFT 0.85). Edge F1은 같은 값에서 y가 −4~+1까지 퍼짐. SSIM>PSNR | 점들이 쌍·블러 내에서 묶여 있어 ρ는 서술용. Repeatability는 검출기를 공유하므로 부분적으로 순환적 |
| F06_predictor_comparison | 4개 지표의 정합 예측력을 세 관점(풀링 ρ / 쌍 내 블러 간 ρ / 셀 내 고정효과 R²)으로 비교 | S3·S4 통계 | 모든 관점에서 Repeatability 최상, Edge F1 최하(쌍 내 ρ 중앙값 0.30–0.33, IQR이 0 근처까지). PSNR·SSIM은 중간 | 세 관점의 수치는 서로 다른 정의라 직접 비교 불가. 어느 지표가 "원인"인지 말하지 않음 |
| F07_heatmap_pair_blur | 쌍×블러별 clean 대비 GT-correct 잔존 비율, A(좌) vs D(우); x = H 실패 | 검출기별 24×8 | A에서 모션 17px 열이 전부 밝음(거의 소실). D는 대부분 진한 파랑(clean 수준 근접)이지만 특정 행(i_castle__1-5, v_artisans__1-5, v_astronautis__1-5, v_underground__1-5, v_cartooncity__1-5)은 D에서도 x가 반복 — clean에서도 실패하는 쌍 | 비율이라 clean 매칭이 적은 쌍(i_castle 등)은 값이 불안정 |
| F08_orientation_asymmetry | clean 영상의 기울기 방향 에너지 비 vs 모션 0°−90° 차이(PSNR, Edge F1, SIFT GT-correct) | 24쌍, 모션 17px, A | PSNR·Edge F1 비대칭은 장면의 엣지 방향으로 거의 완전히 설명(ρ −0.96, −0.98). 매칭 비대칭은 설명되지 않음(ρ −0.30, n.s.) | 24점 상관. 매칭 비대칭의 원인은 미해결 |
| F09_orb_tuning_tradeoff | ORB 조정 설정 − 기본 설정 | 블러별 24쌍 대응 차이 | 강한 블러에서 특징점 +58~+67, ratio 매칭 +14, GT-correct +6.5~7.5; 약한 블러에서는 GT-correct −4; precision은 모두 0 이하 추세(CI 포함 0) | Success 차이는 F02 참조(유의하지 않음). 그리드 밖 설정은 알 수 없음 |
| F10_h_error_ecdf | 그룹별 H 투영 오차의 경험 누적분포(실패는 오른쪽 끝) | 검출기별 192셀(+clean 24) | D 곡선이 clean에 근접(ORB 중앙값 1.06 vs clean 0.80; SIFT 0.63 vs 0.55). A/B/C는 서로 거의 겹침. 미정의(실패) 비율 ORB A 28/192 → D 16/192, SIFT 15 → 1 | 오차 분포의 꼬리는 소수 쌍이 좌우 |
| F11_edge_precision_recall | 블러별 엣지 Precision·Recall의 이동: 블러(점)→Unsharp(주황)→Wiener(청록) | 영상 수준 평균 | 블러는 recall만 깎음(precision ≥0.88). Unsharp는 recall을 올리는 대신 precision을 0.05–0.07 낮춤(가짜 엣지). Wiener는 recall을 0.6–0.84까지 올리면서 precision 0.99 유지 | 평균 이동만 표시. 픽셀 단위 엣지 위치 오차는 보이지 않음 |
| F12_effects_by_sequence_type | 조명(채움)/시점(빈) 시퀀스별 B−A, C−A, D−A | 유형당 12쌍(6 시퀀스), CI는 6 클러스터 부트스트랩 | D−A는 두 유형 모두 양(+0.2~0.3). C−A는 ORB 조명에서만 양(+0.10), SIFT 조명에서는 음(−0.05, CI 상한 0.00). B−A는 양쪽 모두 0 근처 | 6 클러스터라 CI가 넓고 보수적 해석 필요 |
| F13_keypoint_counts | 그룹·블러별 target 특징점 수 | 24쌍 평균 | ORB는 약한 블러에서 1500 예산에 포화(A에서 67–71% 셀), SIFT는 블러 강도에 따라 단조 감소(모션 17px에서 clean의 43%). D는 둘 다 clean 수준 복원 | 특징점 수와 정확도의 관계는 T14·F09 참조 |
| F14_slopegraph_motion17 | 쌍별 GT-correct 경로 A→C→D→clean | 모션 17px 0°, 24쌍, log 축 | C는 A와 거의 같은 높이(평행), D에서 한 자릿수 상승, clean은 D보다 약간 더 높음. 조명/시점 구분 없이 같은 패턴 | 한 블러 조건만 표시 |
| visual_examples/v_busstop__1-2__motion_l17_a90__crops.png | 동일 영상 crop: clean/A/C/D와 Canny 오버레이(초록 TP, 빨강 FP, 파랑 FN) | 모션 17px 90° | C는 세로 줄무늬(링잉)가 증폭되어 PSNR 20.0→19.9, 그러나 Edge F1 0.31→0.56(R 0.19→0.40, 빨간 가짜 엣지 증가). D는 PSNR 27.0, Edge 0.81, GT-correct 35→392(ORB) | 한 쌍의 예시. 다른 쌍의 외형은 다를 수 있음 |
| visual_examples/v_calder__1-5__disk_r5__keypoints.png | clean 특징점의 생존(초록)/소실(빨강) | Disk r5, ORB·SIFT, A/C/D | SIFT A에서 대부분 빨강(rep 0.20), D에서 초록 다수(0.58). ORB는 D에서 예산 1500 복귀 | 반복성 정의는 위치 기반 3px |
| visual_examples/v_calder__1-5__disk_r5__ORB__{A,D}__matches.png | 매칭 선(초록 GT-correct, 빨강 오류)과 H_gt(청록)·H_est(노랑) 외곽선 | ORB A vs D | A: 25 매칭 중 4 정답, H_est 없음(horizon 실패). D: 64 중 44 정답, 노란 외곽선이 청록과 일치 | 한 셀의 예시 |
| visual_examples/v_underground__1-5__none__ORB__CLEAN__matches.png | clean에서도 실패하는 쌍의 ORB 매칭 | clean, ORB | 반복 점무늬 그래피티에서 18 매칭 중 3 정답, H 오차 470 px. SIFT는 같은 쌍에서 44/68 정답, 오차 1.0 px | 블러와 무관한 ORB 기술자의 한계 사례 |

참고: dataviz 검증 스크립트(node)가 설치되어 있지 않아 팔레트 검증기를 다시 실행하지 못했다. 사용한 색은 스킬 문서에 검증 완료로 기록된 참조 팔레트를 그대로 썼고(인접 5슬롯, 산점도는 3계열 이하), 상태색은 쓰지 않았다.
