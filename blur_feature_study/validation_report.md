# Validation Report — 블러 영상 복원 및 구조적 특징 보존성 비교 실험

작성일: 2026-10-10. 이 문서는 구현 검증 결과를 기록하며 연구 가설에 대한 결론을 담지 않는다.

## 0. 요약

| 항목 | 결과 | 근거 파일 |
|---|---|---|
| 단위·통합 테스트 | **54/54 통과** (최종 코드 기준) | `logs/pytest_20261010-003100.txt/.xml` |
| Phase 1 데이터 점검 | 32쌍, H 방향 검증 통과(1쌍은 GT 오류로 판정, 아래 2절) | `data/homography_checks_20261010-002103.*` |
| Phase 2 Pilot | 136행, **32/32 검사 통과** | `results/pilot/pilot_20261010-002906/verification_20261010-002953.md` |
| Phase 3 Validation | 1424행, **29/29 검사 통과**, 파라미터 선택 완료 | `results/validation/validation_20261010-003108/verification_20261010-003436.md` |
| Phase 4 Main | 1632/1632행, 파이프라인 오류 0 | `results/main/main_20261010-003501/raw_results.csv` |
| Phase 5 결과 검증 | **34/34 검사 통과** | `results/main/main_20261010-003501/verification_20261010-004041.md` |
| 교차 실행 재현성 | 272행 × 비시간 컬럼 전부 **완전 일치** | `results/reproducibility/repro_main_20261010-003501_20261010-004109/comparison.json` |

실패한 테스트가 남은 상태에서는 다음 Phase로 진행하지 않았다. 개발 중 실패했던 테스트와 그 원인은 5절에 기록했다.

---

## 1. 단위·통합 테스트 (pytest, 54개)

| 파일 | 개수 | 검증 내용 |
|---|---|---|
| `test_psf.py` | 8 | 8개 PSF의 합=1(|Δ|<1e-12), 비음수, 홀수 크기, 중심 일치, 점대칭. Gaussian 크기(13, 25). Disk의 균일성, 부분 피복, 면적≈πr²(<1%). Motion 방향(0°=수평, 90°=수직). 짝수 길이·사선 각도 거부 |
| `test_imaging.py` | 11 | convolution = `scipy.ndimage.convolve(mode='reflect')`(비대칭 커널, 1e-9). 임펄스 응답 = 중심 PSF. 블러·Wiener의 180° 회전 등변성(이동 없음, 1e-9). Wiener 임펄스 응답 중심·대칭(8 PSF × 3 balance). 반복 LS 이동 추정기 보정(0.3/−0.2px를 ±0.03 이내로 검출). Wiener가 PSNR을 개선하고 이동 < 0.05px(8 PSF × 3 balance). 경계 밴드 오차 ≤ 내부 오차 수준. Unsharp 수식 일치·클리핑 |
| `test_dataset.py` | 8 | 리사이즈 크기 규칙(확대 없음). **S 행렬이 cv2.resize(INTER_AREA) 좌표계와 일치**(서브픽셀 blob 중심, 오차 < 0.02px, 3개 배율). H 리사이즈 점 일관성. Seed 선정 결정성·split 분리. Manifest 일관성. 쌍 로딩 |
| `test_metrics.py` | 17 | PSNR(동일=inf, 알려진 오프셋, 경계 무시). SSIM(동일=1, skimage 일치). Edge F1(동일=1, 2px 이동=1, 3px 이동=0, **가짜 엣지 → Precision만 감소**, 빈 맵 규약). Repeatability(허용오차 경계 2.9/3.1px, **일대일 제약**, greedy가 아닌 최적 매칭, 중복 제거). GT 오차(w<0 → ∞). Homography 평가(오차 0/4/6px → success 및 failure_reason). 실패 처리(None, NaN, 특이 행렬, horizon, 격자 부족). 격자 overlap 제한 |
| `test_features.py` | 8 | 튜닝 grid(ORB 8, SIFT 4, default 1회 포함, nfeatures=1500). ORB/SIFT 검출 **결정성**(키포인트·기술자 비트 일치), 마스크 준수, 예산. 동일 영상 매칭 정확도 > 99%. 빈 기술자 처리. RANSAC이 30% 이상치에서 H 복원. 4점 미만이면 `insufficient_matches` |
| `test_pipeline.py` | 2 | 실제 HPatches 쌍으로 전체 파이프라인을 2회 실행해 **결정성**(비시간 컬럼 일치) 확인. 필수 컬럼, 값 범위. NPZ에서 GT-correct 재계산이 CSV와 일치. 같은 처리 영상을 쓰는 그룹 간 영상 지표 일치. CLEAN 행 PSNR=inf, SSIM=1, F1=1, Rep=1, Success@3=1 |

---

## 2. Phase 1 — 데이터·Homography 검증

- HPatches 압축 파일(1,279,545,572 bytes)은 `zipfile.testzip()` 무결성 검사를 통과했다. 경로 탈출 항목은 없었다. 116개 시퀀스(i_ 57, v_ 59)가 모두 `1–6.ppm`, `H_1_2…H_1_6`를 갖추고 있었고 모든 영상이 읽혔다.
- 선정: Validation 4개 시퀀스(8쌍), Test 12개 시퀀스(24쌍). split 간 시퀀스 중복은 없다.
- **H 방향 검증**(리사이즈 해상도, clean SIFT 매칭, 3px 이내 매칭 수):
  - viewpoint 16쌍: `H`는 55–2400개, `inv(H)`는 0–4개, 항등은 0–7개. **H_1_k = Reference → Target**이 확인되었다.
  - illumination 쌍: H ≈ I라 방향 판별력은 없지만, 일관성(H 기준 올바른 매칭 다수)을 확인했다.
- **리사이즈 규약**:
  - 단위 테스트로 픽셀 중심 S가 cv2.resize와 일치함을 확인했다(오차 < 0.02px).
  - HPatches H 자체의 규약은 원해상도 viewpoint 39쌍에서 확인했다. H를 그대로 쓴 경우(평균 중앙 오차 0.78px)가 corner→center 변환(0.87px)보다 33/39쌍에서 우수했다. 따라서 H를 픽셀 중심 규약으로 사용한다.
- **발견된 데이터 문제 — `i_londonbridge` GT 부정확** (validation 시퀀스):
  - H_1_k가 모두 항등행렬이지만 clean 영상의 RANSAC inlier 이동량 중앙값은 원해상도에서 t=2: (0.7, 3.84)px, t=5: (2.43, 5.48)px였다. 촬영 중 카메라가 움직인 것으로 보이며, 파이프라인 오류가 아니라 데이터셋 GT 문제다.
  - 사전 선언한 clean-영상 규칙(올바른 매칭 < 10 또는 근접 매칭 중앙 오차 > 1.5px)으로 `gt_quality_flag`를 부여했다: `i_londonbridge__1-2`, `i_londonbridge__1-5`, `v_cartooncity__1-5`(test, 1.72px).
  - 쌍은 제거하지 않았다(seed 선정 보존). 플래그 쌍을 제외해도 Validation 파라미터 선택은 같았다.

---

## 3. Phase 2 — Pilot

- 대상: 사전 선언 규칙 "유형별로 GT 플래그가 없는 첫 Validation 시퀀스의 target 2"에 따라 `i_veggies__1-2`, `v_birdwoman__1-2`. 8개 블러 조건 전체, A/B/C/D/CLEAN.
  - B는 임시 설정(ORB sf1.1/nl12/ft10, SIFT ct0.02/s1.2), D는 임시 balance 0.02를 썼고 `is_placeholder=1`로 표시했다.
- 명세의 9개 확인 항목에 대응하는 32개 검사가 모두 통과했다. 주요 항목:
  1. PSF 정규화·중심(8조건)
  2. Unsharp/Wiener 출력 형식, 실제 영상에서 Wiener 회전 등변성(maxdiff 1.3e-15)
  3. 처리 영상 크기 = clean target 크기
  4. PSNR을 `cv2.PSNR`로, SSIM을 skimage로 **처리 영상 48개 전부** 재계산(최대 차이 7e-15 dB / 1e-16)
  5. Edge F1·Repeatability 범위, CLEAN=1
  6. 키포인트 존재
  7. NPZ에서 GT-correct 재계산 일치
  8. H_est로 격자 오차·성공·실패 원인 재계산 일치
  9. CSV 필수 컬럼·행 수(136)·중복 없음
  - 상식 검사: 블러 강도가 커지면 PSNR이 감소. Clean 기준선이 두 pilot 쌍 × 두 검출기 모두 @3px 성공(오차 0.06–0.62px).
- QC 영상 42장(`qc/pilot_20261010-002906/`)을 육안으로 확인했다. 모션 방향(0°=수평 번짐), Wiener 복원 선명도, H_est와 H_gt 외곽선의 일치를 보았다.

---

## 4. Phase 3 · 4 · 5

### 4.1 Validation (1424행)
- 29개 검사 통과. 파라미터 선택은 `config/experiment_config.yaml`의 `selection_rules`(실행 전 작성)를 그대로 적용했다. 결과는 `selection_detectors.csv`, `selection_wiener.csv`, `selected_params.yaml`에 있다.
  - ORB → sf1.2 / nl12 / ft20 (Success@3px 0.672 vs 기본 0.641)
  - SIFT → 기본 설정 (0.641 vs 차순위 0.625)
  - Wiener → 8조건 모두 0.005

### 4.2 Main 처리 건수 (Test 24쌍)

| 그룹 | 검출기 | 설정 | 블러 조건당 | 합계 |
|---|---|---|---|---|
| A, B, C, D | ORB, SIFT | default / tuned / default / default | 24 | 각 192 (8조건) |
| CLEAN | ORB, SIFT | default (+tuned 보조) | — | 각 24 (+24) |
| **총계** | | | | **1632** (예상과 일치, 누락·중복 0) |

8개 지표의 계산 여부(비NaN 행 수):
- PSNR·SSIM·Edge F1·Repeatability·GT-correct·Match Precision·Success: 모든 행(1632/1632)에서 계산되었다. 매칭 0개인 행은 없었다.
- Homography Projection Error: H 추정 실패 행에서는 설계대로 NaN이다(145행). 그 외 1487행에서 계산되었다.

`failure_reason` 분포(Main, Success@3px 기준):

| 그룹/검출기 | 성공@3 | 오차 3–5px | 오차 >5px | invalid: horizon | invalid: singular | 매칭 <4 |
|---|---|---|---|---|---|---|
| A ORB | 91 | 24 | 49 | 27 | 1 | 0 |
| A SIFT | 133 | 16 | 28 | 9 | 6 | 0 |
| B ORB | 98 | 22 | 45 | 18 | 9 | 0 |
| B SIFT | 133 | 16 | 28 | 9 | 6 | 0 |
| C ORB | 107 | 16 | 42 | 21 | 5 | 1 |
| C SIFT | 136 | 19 | 25 | 6 | 6 | 0 |
| D ORB | 141 | 13 | 22 | 12 | 4 | 0 |
| D SIFT | 174 | 11 | 6 | 1 | 0 | 0 |
| CLEAN ORB (48) | 36 | 1 | 7 | 4 | 0 | 0 |
| CLEAN SIFT (48) | 46 | 0 | 2 | 0 | 0 | 0 |

### 4.3 Phase 5 결과 검증 (34개 검사, 모두 통과)
- **결과 행 수와 조건**: 1632 = 예상. 키 중복 0, 누락·초과 0. 조건별 24쌍으로 균형.
- **Split 위생**: Main 쌍은 모두 Test split에서 나왔다. B 행의 `detector_params`와 D 행의 `wiener_balance`가 Validation 선택 파일과 일치하고, 선택 파일 출처는 `validation_20261010-003108`이다.
- **같은 평가 규칙**: 모든 행이 단일 `config_hash`(3315ea402938751e)와 `code_hash`(e34adc886e87d110)를 가진다. margin 24, 예산 1500 모두 동일.
- **PSF**: 8조건 정상(합 1.000000000000).
- **H 방향·리사이즈**: NPZ에 저장된 H_gt가 manifest의 H_resized와 1e-12 이내로 일치한다. 처리 영상 크기 = 리사이즈된 target 크기.
- **PSNR·SSIM 기준**: 처리 영상 576개 **전부**를 재생성해 `cv2.PSNR`(독립 구현)과 skimage SSIM으로 재계산했다. 최대 차이 1.4e-14 dB / 1.1e-16. CLEAN은 PSNR=inf, SSIM=1. data_range=255.
- **Edge F1·Repeatability 범위**: [0,1], NaN 없음. 576개 영상에서 Edge F1 재계산이 일치.
- **GT-correct 오차 계산**: 1632개 NPZ 전부에서 좌표와 H_gt로 재계산한 @3/@5 개수, 매칭 수, inlier 수가 일치한다. 인덱스와 좌표가 키포인트 배열과 일치한다.
- **Homography 실패 처리**: NaN 오차 ⇔ success=0이며 failure_reason이 있다. success@5 ≥ success@3. NPZ의 H_est로 오차·성공·실패 원인을 재계산해 1632행 모두 일치.
- **이동 없음**: 실제 test 영상에서 Wiener 회전 등변성 maxdiff 2.0e-15.

---

## 5. 발견된 오류 및 수정 내역

| # | 단계 | 문제 | 원인 | 조치 |
|---|---|---|---|---|
| 1 | Phase 1 | `cv2.imread`가 모든 영상에서 실패 | Windows에서 OpenCV가 비ASCII(한글 사용자명) 경로를 열지 못함 | 모든 영상 I/O를 `np.fromfile + cv2.imdecode` / `imencode + tofile`로 교체 (`io_utils.py`) |
| 2 | Phase 1 | `i_londonbridge` H 방향 검사 실패 | 데이터셋 GT 부정확(카메라 이동, 2절) | 파이프라인 변경 없음. 사전 선언 규칙으로 GT 품질 플래그 부여, 쌍은 유지 |
| 3 | 테스트 | "이동 없음" 테스트 4건 실패(0.02–0.11px) | `cv2.phaseCorrelate`는 크기 정보를 버린다. OTF≈0 또는 음수인 주파수에서 양자화 잡음이 위상을 흔들어 서브픽셀 피크가 편향됨. **추정기 문제이고 실제 이동이 아님**: float 블러에서도 같은 값이 나왔고, 블러가 정확히 회전 등변(1e-13)임을 확인 | 정확한 검증으로 교체: ① 블러+Wiener 임펄스 응답의 중심·대칭 ② 180° 회전 등변성(이동 s가 있으면 2s 차이로 드러남) ③ 보정된 반복 LS 이동 추정기(Wiener 출력에만 사용) |
| 4 | 테스트 | LS 이동 추정기 보정 실패(0.3px를 0.37px로 추정) | 1회 선형화의 편향 | 반복(Gauss–Newton) 정합으로 개선. 보정 테스트를 추가해 0.03px 이내 검출 확인 |
| 5 | 테스트 | 강한 방향성 블러(motion 17px)에서 LS 추정기가 0.07px로 읽음 | 유한 창·비대칭 내용에서 생기는 추정기 편향(양자화 없는 float에서도 동일) | 블러 영상의 이동 검증은 회전 등변성(정확한 검사)으로 대체 |
| 6 | 테스트 | 영행렬 H 검사에서 RuntimeWarning(0/0) | 정규화 전에 norm=0을 확인하지 않음 | norm=0이면 즉시 singular로 판정 |
| 7 | 설계 | skimage `wiener(clip=True)` 기본값 | 출력을 [-1,1]로 자름 | `clip=False`, 입력 [0,1], 출력 클리핑과 uint8 양자화를 직접 수행 |

---

## 6. 재현성 검증

- **프로세스 내 결정성**: 같은 쌍과 조건으로 파이프라인을 2회 실행한 결과, 비시간 컬럼이 모두 일치했다(`test_pipeline.py`). ORB/SIFT 키포인트·기술자도 비트 단위로 일치했다(`test_features.py`).
- **교차 실행 재현성**: 새 프로세스에서 Test 4쌍(`i_gonnenberg__1-2/1-5`, `v_busstop__1-2/1-5`) × 9조건 = 272행을 다시 실행했다. 원본 main 결과와 비교한 결과 `runtime_*`, `timestamp`, 경로, `code_hash`를 제외한 모든 컬럼(지표, 키포인트 수, 매칭 수, H 오차, 실패 원인)이 **완전 일치**했다(불일치 0).
- **재계산 일관성**: 4.3절의 독립 재계산(PSNR/SSIM 576개 영상, NPZ 1632개)이 모두 일치했다.
- 난수 시드: 42(시퀀스 선정). RANSAC은 `cv2.setRNGSeed(42)`를 호출한 뒤 OpenCV의 결정적 RNG를 사용한다.

## 7. 실험 해석 시 유의할 검증상 사실 (결론 아님)
- SIFT의 Validation 선택 설정이 기본값과 같아서 Test SIFT B 행은 A와 192/192 동일하다(B−A ≡ 0).
- Wiener balance는 8조건 모두 후보 최솟값(0.005)으로, grid 경계다.
- Clean 기준선 실패(@3px): ORB 6쌍, SIFT 1쌍. `v_artisans__1-5`는 clean에서도 두 검출기 모두 @5px 실패다.
- ORB D(Wiener)의 Success@5px(0.802)가 CLEAN-default ORB(19/24 = 0.79)와 비슷하거나 약간 높다. 복원 영상의 아티팩트성 특징이 기여하는지 개별 쌍 수준에서 확인할 필요가 있다.
