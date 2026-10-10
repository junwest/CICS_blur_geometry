# 블러 영상 복원 및 구조적 특징 보존성 비교 실험

**국문:** 블러 영상 복원에 따른 구조적 특징 보존성과 기하학적 정합 성능 분석
**영문:** Analysis of Structural Feature Preservation and Geometric Matching Performance in Deblurred Images

이 저장소는 새로운 복원 알고리즘을 제안하지 않는다. 영상 복원(Unsharp Mask, Known-PSF Wiener)이 ORB·SIFT의 검출 파라미터 조정과 비교하여 유효한 특징 정보를 추가로 제공하는지 측정하기 위한 **재현 가능한 비교 실험**이다. 현재 단계의 산출물은 원시 결과이며, 가설(H1–H3)에 대한 결론은 아직 작성하지 않았다.

- RQ1: PSNR·SSIM 향상이 Edge F1·Keypoint Repeatability 향상과 일치하는가?
- RQ2: 엣지·특징점 보존 개선이 GT 기준 올바른 매칭 수와 Homography 정확도 개선으로 이어지는가?
- RQ3: 검출 파라미터 조정과 비교해 영상 복원이 추가적인 개선을 제공하는가?

---

## 1. 실행 결과 요약 (2026-10-10)

| 단계 | run_id | 결과 |
|---|---|---|
| Phase 1 데이터 점검 | `data/homography_checks_20261010-002103.*` | 32쌍 manifest, H 방향·리사이즈 검증, GT 품질 플래그 3쌍 |
| 단위/통합 테스트 | `logs/pytest_20261010-003100.*` | 54/54 통과 |
| Phase 2 Pilot | `pilot_20261010-002906` | 136행, 검증 32/32 통과 |
| Phase 3 Validation | `validation_20261010-003108` | 1424행, 검증 통과, 파라미터 선택 완료 |
| Phase 4 Main (Test) | `main_20261010-003501` | 1632행, 파이프라인 오류 0 |
| Phase 5 결과 검증 | `verification_20261010-004041.md` | 34/34 통과 |
| 교차 실행 재현성 | `results/reproducibility/repro_main_…_20261010-004109` | 272행 전 컬럼 일치 |

세부 내역은 [validation_report.md](validation_report.md) 참조.

---

## 2. 디렉터리 구조

```text
blur_feature_study/
├── config/
│   ├── experiment_config.yaml                  # 모든 설정 + 사전 등록된 선택 규칙
│   └── selected_params_validation_<id>.yaml    # Validation에서 선택된 ORB/SIFT/Wiener 파라미터
├── data/
│   ├── hpatches-sequences-release/             # HPatches 원본 전체 영상 시퀀스 (116개)
│   ├── hpatches_download/…zip                  # 원본 압축 파일 (보존)
│   ├── dataset_manifest.csv                    # 선정 시퀀스·파일·H·분할 정보
│   ├── gt_quality_flags.csv                    # GT 품질 플래그 (clean 영상만으로 산출)
│   ├── homography_checks_<ts>.csv/json         # H 방향·리사이즈 검증
│   └── pixel_convention_check_<ts>.csv         # HPatches 픽셀 좌표 규약 검증
├── src/                                        # 실험 코드 (아래 3절)
├── tests/                                      # pytest 단위·통합 테스트 (54개)
├── results/
│   ├── pilot/<run_id>/                         # Phase 2
│   ├── validation/<run_id>/                    # Phase 3 (+ selection_*.csv, selected_params.yaml)
│   ├── main/<run_id>/                          # Phase 4 (raw_results.csv)
│   ├── matching_details/<run_id>/<pair_id>/*.npz   # 개별 매칭·H_gt·H_est
│   └── reproducibility/<id>/comparison.json
├── qc/<run_id>/                                # 코드 검증용 예시 영상 (논문 Figure 아님)
├── logs/                                       # 실행 로그, pytest 결과
├── environment.txt, requirements.txt, README.md, validation_report.md
```

각 run 디렉터리에는 `snapshot/`(당시 src, config, manifest, 선택 파라미터 사본), `environment.txt`, `run_meta_attempt*.json`, `raw_results.csv`, `raw_results_all_attempts.csv`, `errors_*.csv`, `verification_*.md/json`이 저장된다. 기존 파일은 덮어쓰지 않는다(run 디렉터리 재사용 금지, NPZ·보고서 덮어쓰기 거부).

---

## 3. 파이프라인

```text
HPatches 시퀀스 → (seed 42) 16개 선정 → Reference 1 / Target 2, 5
   → 그레이스케일 → 긴 변 960px 리사이즈(INTER_AREA, 확대 없음), H_resized = S_t·H·S_r⁻¹
   → Target에만 블러(8조건) → uint8 양자화
   → 처리: Blurred | Unsharp Mask | Wiener(Known PSF)
   → ORB/SIFT 검출(공통 경계 마스크) → Clean Reference ↔ Processed Target 매칭(BF, KNN k=2, ratio 0.75)
   → RANSAC Homography → 8개 지표 + 매칭 세부 저장
```

| 모듈 | 역할 |
|---|---|
| `src/io_utils.py` | 유니코드 경로 영상 I/O(np.fromfile+imdecode), 설정 로드, 해시, 덮어쓰기 방지 |
| `src/dataset.py` | 시퀀스 선정, manifest, 리사이즈와 H 변환 |
| `src/psf.py` | Gaussian / Disk / Motion PSF |
| `src/imaging.py` | 블러 생성, Unsharp Mask, Wiener |
| `src/features.py` | ORB/SIFT 생성·검출, ratio 매칭, RANSAC |
| `src/metrics.py` | 8개 평가지표 |
| `src/pipeline.py` | 실험 계획(그룹) 생성, 행 단위 평가, NPZ 저장, CSV 기록 |
| `src/run_experiment.py` | Phase 실행 CLI (pilot / validation / main, resume) |
| `src/select_params.py` | 사전 등록 규칙에 따른 Validation 파라미터 선택 |
| `src/verify_results.py` | 결과 무결성 검증 + QC 영상 |
| `src/repro_check.py` | 교차 실행 재현성 검증 |
| `src/prepare_data.py` | Phase 1 (manifest, H 방향, 픽셀 규약, GT 품질 플래그) |
| `src/env_report.py` | environment.txt |

### 3.1 데이터
- HPatches full image sequences (`hpatches-sequences-release`, 116개: i_ 57, v_ 59). 65×65 패치가 아닌 원본 전체 영상을 사용한다.
- 선정: `rng = numpy.random.default_rng(42)`로 i_ 정렬 목록을 permutation해 8개, 이어서 같은 rng로 v_ 목록에서 8개를 뽑는다. 각 유형의 앞 2개는 Validation, 나머지 6개는 Test다(시퀀스 단위 분리).
  - Validation: `i_londonbridge, i_veggies, v_birdwoman, v_apprentices` (8쌍)
  - Test: `i_gonnenberg, i_kurhaus, i_castle, i_fruits, i_lionnight, i_steps, v_busstop, v_cartooncity, v_calder, v_astronautis, v_artisans, v_underground` (24쌍)
- 쌍: Reference `1.ppm`, Target `2.ppm`, `5.ppm`, GT `H_1_2`, `H_1_5`.
- **H 방향**: `H_1_k`는 Reference(1) 좌표를 Target(k) 좌표로 보낸다. clean SIFT 매칭으로 검증했다. viewpoint 쌍에서 H는 수백~수천 개, inv(H)는 0~4개의 올바른 매칭을 보였다.
- **리사이즈**: 영상별 `S = [[sx,0,0.5sx−0.5],[0,sy,0.5sy−0.5],[0,0,1]]`(픽셀 중심 규약, cv2.resize 좌표계와 일치하며 단위 테스트로 검증), `H_resized = S_target · H_original · inv(S_reference)`, H[2,2]=1로 정규화한다. 10개 viewpoint 시퀀스는 Reference와 Target 크기가 달라 영상별 S를 사용한다.
- HPatches H의 픽셀 규약: 원해상도에서 H를 그대로 쓴 경우가 corner→center 변환보다 39쌍 중 33쌍에서 오차가 작았다. 따라서 H는 픽셀 중심 규약으로 취급한다.
- GT Homography는 데이터셋이 제공하는 **추정 기준값**이며 완전한 기하학적 정답이 아니다. 새로 추정한 H를 정답으로 쓰지 않는다.

### 3.2 블러 (8조건, Target에만 적용)
| blur_id | 종류 | 파라미터 | 커널 |
|---|---|---|---|
| gaussian_s1.5 / gaussian_s3.0 | Gaussian | σ=1.5 / 3.0, 반경 ceil(4σ) | 13×13 / 25×25 |
| disk_r3 / disk_r5 | Disk defocus | 반경 3 / 5 px, 33×33 서브샘플링 면적 가중(anti-aliased) | 7×7 / 11×11 |
| motion_l9_a0, _a90 / motion_l17_a0, _a90 | 선형 모션 | 길이 9 / 17 px, 0°=수평, 90°=수직 | 9×9 / 17×17 |

- 모든 PSF는 합 1, 비음수, 홀수 크기, 중심 일치, 점대칭이다(자동 테스트).
- 공통 경계 규칙: half-sample symmetric reflect(`cv2.BORDER_REFLECT` = numpy `symmetric` = scipy `reflect`). 진짜 convolution(커널 flip)이며 scipy와 1e-9 이내로 일치한다.
- 블러 영상은 반올림·클리핑 후 uint8로 저장한다. 추가 노이즈는 없고 8-bit 양자화가 유일한 잡음원이다.

### 3.3 비교 실험군
| 그룹 | 영상 처리 | 검출기 |
|---|---|---|
| CLEAN | Clean target | 기본 (+ 조정 설정, `is_supplementary=1`) |
| A | Blurred | 기본 |
| B | Blurred | Validation에서 선택한 조정 설정 |
| C | Unsharp Mask `I + α(I − G_σ*I)`, α=1.0, σ=1.2, [0,255] 클리핑 (단순 선명화로 분류) | 기본 |
| D | Wiener, **Known-PSF / Oracle-PSF** (blind deblurring 아님) | 기본 |

- 검출기 "기본" 설정은 OpenCV 기본값에 `nfeatures=1500`만 고정한 것이다. 모든 그룹의 특징점 예산을 같게 하기 위해서다.
  - ORB: scaleFactor 1.2, nlevels 8, edgeThreshold 31, firstLevel 0, WTA_K 2, HARRIS, patchSize 31, fastThreshold 20
  - SIFT: nOctaveLayers 3, contrastThreshold 0.04, edgeThreshold 10, sigma 1.6, enable_precise_upscale False
- Wiener: `skimage.restoration.wiener`(Wiener–Hunt, Laplacian 정규화), `X = H*·Y / (|H|² + balance·|L|²)`. 입력은 [0,1] float이고 `clip=False`(skimage 기본값 clip=True는 [-1,1]로 자르므로 사용하지 않음), 출력은 uint8. FFT 전에 2H×2W 대칭 확장을 한다. 대칭 PSF와 reflect 경계 블러에서는 이것이 정확한 순환 합성곱 모델이 되므로, 경계 링잉과 영상 이동이 발생하지 않는다(임펄스 응답·회전 등변성 테스트로 검증).
- 공통 유효 평가 영역: 모든 경계에서 24px 이상 떨어진 영역이다(최대 PSF 반경 12 + unsharp 반경 5 + 허용오차 3 + Sobel/NMS 2 = 22 → 24). 같은 경계 마스크를 Reference, Clean target, Processed target의 **특징점 검출 마스크**로도 사용한다.

### 3.4 Validation 파라미터 선택 (사전 등록 규칙, Validation 8쌍만 사용)
- **검출기(B)**: 검출기별로 하나의 고정 설정을 고른다. 8쌍 × 8블러 = 64케이스를 합쳐 ① 평균 Success@3px 최대, ② 중앙값 투영 오차 최소(실패 = +∞), ③ 평균 GT-correct@3px 최대, ④ 기본 설정 우선, 이후 grid 순서.
  - ORB 후보 8개 = scaleFactor {1.1,1.2} × nlevels {8,12} × fastThreshold {10,20}
  - SIFT 후보 4개 = contrastThreshold {0.02,0.04} × sigma {1.2,1.6}
- **Wiener balance(D)**: 블러 조건별로 {0.005, 0.02, 0.08} 중 ① Validation 평균 PSNR 최대, ② SSIM, ③ 더 큰 balance.
- **선택 결과** (`config/selected_params_validation_20261010-003108.yaml`):
  - ORB → `scaleFactor 1.2, nlevels 12, fastThreshold 20`. Success@3px 0.672 vs 기본 0.641로 ①에서 결정.
  - SIFT → **기본 설정과 동일**(0.04, 1.6). Success@3px 0.641 vs 차순위 0.625로 ①에서 결정. 따라서 **Test의 SIFT B는 A와 192행 모두 동일**하다.
  - Wiener → 8개 조건 모두 **0.005**(후보 grid의 최솟값).
  - GT 플래그 쌍(i_londonbridge)을 제외해도 선택 결과는 같다.
- Test 데이터로는 어떤 파라미터도 조정하지 않았다(검증 스크립트가 split과 선택 파라미터 출처를 확인).

---

## 4. 평가지표 정의

Clean target을 `I`, 처리된 target을 `Î`, 유효 영역을 `Ω`(경계 24px 제외)라 한다.

1. **PSNR** = `10·log10(255² / mean_Ω (I − Î)²)`. uint8, data_range=255. 동일 영상이면 `inf`(CLEAN 행).
2. **SSIM**: skimage `structural_similarity`, Ω로 크롭한 영상, data_range=255, Gaussian window σ=1.5, `use_sample_covariance=False`, K1=0.01, K2=0.03 (Wang et al. 2004 설정).
   PSNR·SSIM은 항상 같은 Target의 Clean/Processed 쌍에서 계산하며, Reference와 Target 사이에서는 계산하지 않는다.
3. **Edge F1**: Canny(80, 160, L2gradient=True, aperture 3), 동일 설정. `E`, `Ê`는 엣지 맵, `d_E(x)`는 가장 가까운 E 엣지까지의 유클리드 거리(distance transform)다. Ω 안의 엣지 픽셀만 센다.
   - Precision = |{x ∈ Ê∩Ω : d_E(x) ≤ 2}| / |Ê∩Ω|
   - Recall = |{x ∈ E∩Ω : d_Ê(x) ≤ 2}| / |E∩Ω|
   - F1 = 2PR/(P+R). 일대일 제약이 없는 허용오차 방식이며, 가짜 엣지가 늘면 Precision만 감소한다(테스트로 확인).
   - 규약: Clean 엣지 0개이면 P, R, F1 모두 NaN. 처리 엣지 0개이면 P=NaN, R=0, F1=0.
4. **Keypoint Repeatability** (위치 기반 단순화 정의) = `N_repeated / N_clean_valid`. 같은 영상 좌표계에서 Clean target의 특징점과 처리 영상의 특징점을 비교하며, 같은 검출기와 같은 설정을 쓴다.
   - 허용오차 3px, 일대일 대응(허용오차 그래프의 연결 성분마다 Hungarian 최대 매칭).
   - 동일 (x,y) 중복점(예: SIFT 다중 방향)은 1개로 센다.
   - 스케일·영역 중첩 기반의 엄밀한 repeatability(Mikolajczyk et al.)와 다르다. Reference–Target 매칭과도 무관하다.
   - 추가 컬럼 `keypoint_repeatability_vs_default_clean`: 처리 영상 특징점을 **기본 설정** Clean 특징점과 비교한 값(B의 해석용, 그 외 그룹은 주 지표와 같음).
5. **GT-Correct Match Count**: ratio test를 통과한 매칭 i에 대해 `e_i = ‖x_tgt,i − π(H_gt·x_ref,i)‖₂`. 올바른 매칭은 e_i ≤ 3px, 보조로 ≤ 5px. 투영 불가(w ≤ 0)이면 e_i = ∞.
6. **Match Precision** = GT_correct / good_matches(@3, @5). 매칭이 0개이면 NaN이고 correct = 0. RANSAC inlier ratio(`ransac_inliers / good_matches`)와는 별개의 값이다.
7. **Homography Projection Error**: RANSAC(3px, maxIters 2000, confidence 0.995, cv2.RANSAC)으로 추정한 H_est와 H_gt를 비교한다.
   - Reference 유효 영역에 10×10 격자를 놓고, H_gt 투영이 Target 유효 영역에 들어오는 점만 쓴다(`n_grid_valid`).
   - `‖π(H_est·p) − π(H_gt·p)‖`의 평균·중앙값(+최대)을 저장한다. RANSAC 잔차는 정확도 평가에 쓰지 않는다.
8. **Homography Success@3px / @5px** = H_est가 유효하고 평균 격자 오차 ≤ 임계값.
   - 실패 시 오차는 NaN, success는 0이며 행은 삭제하지 않는다.
   - `failure_reason`(Success@3px 기준): `insufficient_matches`(<4), `homography_estimation_failed`, `invalid_homography_nonfinite|singular|horizon_in_overlap`(격자 위에서 w의 부호가 바뀜), `insufficient_grid_points`(<10), `projection_error_gt_3px`(3<e≤5), `projection_error_gt_5px`(e>5), `no_reference_keypoints|no_target_keypoints`, `pipeline_error`.

---

## 5. 실행 방법 / 재현

PowerShell, `blur_feature_study/`에서 실행한다:

```powershell
.\.venv\Scripts\Activate.ps1          # 또는: python -m venv .venv; pip install -r requirements.txt
python -m src.prepare_data            # Phase 1: manifest(동일하면 유지), H 검증, GT 플래그
python -m pytest tests -v             # 54개 테스트
python -m src.run_experiment --phase pilot
python -m src.verify_results --run results/pilot/<run_id> --sample-images 0 --qc
python -m src.run_experiment --phase validation
python -m src.verify_results --run results/validation/<run_id>
python -m src.select_params --run results/validation/<run_id>
python -m src.run_experiment --phase main --selected-params config/selected_params_<validation_run_id>.yaml
python -m src.verify_results --run results/main/<run_id> --sample-images 0 --qc
python -m src.repro_check --run results/main/<run_id> --n-per-type 2
```

- 일부 조건만 재실행: `--pairs i_castle__1-2 --blurs disk_r5`. 중단된 run이나 실패 행 재시도는 `--resume <run_id>`로 한다. pipeline_error가 없는 행은 건너뛰고, 재시도 행은 `attempt`가 증가한다. `raw_results.csv`는 키별 최신 시도로 다시 만들어지고, 이전 파일은 `raw_results.superseded_<ts>.csv`로 보존된다.
- 결정성: 같은 입력이면 모든 비시간 컬럼이 비트 단위로 같다(교차 실행 272행 확인). `runtime_*`만 달라진다.
- 결과를 다른 환경에서 읽을 때: `pd.read_csv(path)`로 읽는다. `psnr`의 `inf`(CLEAN 행)와 `nan`은 pandas가 float로 해석한다. 매칭 세부는 `np.load(match_detail_file)`로 연다(경로는 프로젝트 루트 기준 상대 경로).

### 5.1 `raw_results.csv` 컬럼 사전 (한 행 = 영상 쌍 × 블러 조건 × 처리 × 검출기 × 설정)

| 컬럼 | 의미 |
|---|---|
| run_id, phase, split | 실행 ID, 단계, validation/test |
| sequence_id, sequence_type, pair_id, reference_index, target_index | 식별자 (`pair_id = <seq>__1-<t>`) |
| blur_id, blur_type, blur_strength, blur_angle | 블러 조건 (strength = σ / 반경 / 길이, angle은 모션에만 있음) |
| group | CLEAN / A / B / C / D (validation: B_candidate, D_candidate) |
| processing_method | clean / blurred / unsharp_mask / wiener_known_psf |
| wiener_balance | D 행의 balance (그 외 빈칸) |
| detector, detector_config, detector_params | ORB/SIFT, default/tuned/grid 이름, 실제 파라미터 JSON |
| is_supplementary, is_placeholder | CLEAN+tuned 보조 행 / pilot 임시 설정 |
| gt_quality_flag | GT 품질 플래그(1 = 주의, 6절) |
| psnr, ssim, edge_precision, edge_recall, edge_f1, edge_n_clean, edge_n_proc | 영상·엣지 지표 |
| keypoint_repeatability, rep_n_repeated, rep_n_clean_valid, keypoint_repeatability_vs_default_clean | 반복성 |
| reference_keypoints, target_keypoints, good_matches | 특징점·ratio 통과 매칭 수 |
| gt_correct_matches_3px/5px, match_precision_3px/5px | GT 기준 매칭 정확도 |
| ransac_inliers, ransac_inlier_ratio, ransac_inliers_gt_correct_3px | RANSAC 결과 |
| homography_error_mean/median/max, n_grid_valid, homography_success_3px/5px | 정합 평가 |
| failure_reason, pipeline_error | 실패 원인 (빈칸 = Success@3px) |
| runtime_ms | 처리 + target 검출 + 매칭 + RANSAC (Reference 검출은 캐시라 제외) |
| runtime_processing_ms / detect / match / homography / reference_detect | 세부 시간 |
| match_detail_file | NPZ 경로: `H_gt, H_est, ref_kp, tgt_kp (x,y,size,angle,response,octave), query_idx, train_idx, dist1, dist2, ref_xy, tgt_xy, gt_err, ransac_inlier, ref_shape, tgt_shape, meta` |
| image_h, image_w, valid_margin_px, config_hash, code_hash, attempt, timestamp | 재현 정보 |

차이 분석(B−A, C−A, D−A)은 `(pair_id, blur_id, detector)`를 키로 그룹 간 조인한다. 같은 쌍에서 나온 8개 블러 조건은 독립 표본이 아니다. 같은 sequence_id의 두 쌍(target 2, 5)도 독립이 아니다.

---

## 6. 제한 사항 및 알려진 문제

1. **Known-PSF(Oracle) 복원**: D는 블러 생성에 쓴 정확한 PSF를 사용한다. blind deblurring이나 실제 PSF 추정 성능으로 해석할 수 없다.
2. **합성 블러**: 공간 불변, 무잡음(8-bit 양자화만 존재)이며 대칭 PSF다. 실제 카메라 블러(공간 가변, 노이즈, 비대칭 모션)와 다르다.
3. **GT 품질**: `i_londonbridge`(validation)는 GT가 항등행렬이지만 실제 영상은 원해상도에서 (0.7, 3.8)px(t=2), (2.4, 5.5)px(t=5) 어긋나 있다. 데이터셋 GT의 부정확성이다.
   - 사전 선언 규칙(clean SIFT 매칭 기준: 올바른 매칭 < 10 또는 근접 매칭 중앙 오차 > 1.5px)으로 `i_londonbridge__1-2`, `i_londonbridge__1-5`, `v_cartooncity__1-5`(test, 1.72px)에 `gt_quality_flag=1`을 부여했다.
   - 이 쌍들은 제거하지 않았다(seed 선정 유지). 분석 시 포함/제외 민감도 분석을 권장한다.
4. **SIFT 조정 = 기본**: Validation 규칙상 SIFT 조정 설정이 기본값과 같아서 Test의 SIFT B−A는 정의상 0이다. 검출기 조정 효과(RQ3)는 ORB에서만 0이 아닌 값으로 관찰된다. Validation 차이도 1/64 케이스로 작았다.
5. **Wiener balance가 grid 경계에 위치**: 8개 조건 모두 최소 후보 0.005가 선택되었다. PSNR 기준 최적값은 더 작을 수 있다. 후보 grid는 실험 명세를 그대로 따랐다.
6. **Clean 기준선 실패**: Test에서 CLEAN-default가 Success@3px에 실패한 경우가 ORB 6쌍, SIFT 1쌍이다(주로 viewpoint target 5). `v_artisans__1-5`는 clean에서도 두 검출기 모두 @5px 실패다. 이런 쌍은 처리 조건 간 차이 분석에서 바닥 효과를 만든다.
7. **Repeatability**는 위치 기반 단순화 정의다(스케일·영역 중첩 미사용). 특징점 예산(1500) 상한 때문에 검출 수가 많은 조건에서 값이 커질 수 있다.
8. **GT-correct는 투영 기반**이다. 저텍스처나 반복 패턴에서는 기술자상 잘못된 매칭이 우연히 3px 안에 들어올 수 있다.
9. **Homography는 단일 평면 정합**이다. 3D 카메라 자세 추정 정확도로 해석하지 않는다.
10. **실행 시간**: CPU 단일 프로세스(OpenCV 내부 멀티스레드 16)에서 측정했다. Wiener 시간(~190ms)에는 2×2 대칭 확장 FFT(float64)가 포함된다. 다른 작업과 동시 실행 시 잡음이 있다.
11. 환경 표기: Python `platform`은 Windows 11(build 26300)을 "Windows-10"으로 표기한다. venv 경로의 한글 사용자명이 일부 출력에서 깨져 보이지만 동작에는 영향이 없다. 한글 경로 때문에 `cv2.imread`를 쓰지 않는다.
12. `code_hash`: validation과 main은 같은 코드(`e34adc886e87d110`)다. pilot은 그 이전 버전(선택·검증 스크립트 추가 전, 실험 로직 동일)이다. `repro_check.py`는 main 이후에 추가되었다.

## 7. 사용 라이브러리
Python 3.11.9, OpenCV(contrib) 5.0.0, NumPy 2.4.6, SciPy 1.17.1, scikit-image 0.26.0, pandas 3.0.6, PyYAML 6.0.3, pytest 9.1.1. 전체 목록은 `requirements.txt`, `environment.txt`에 있다. GPU는 사용하지 않는다.
