# DECISIONS — 확장 실험 판단 기록

각 항목: 결정 / 이유 / 시점. "결과 확인 전"은 해당 실험의 매칭·정합 결과를 하나도 보기 전이라는 뜻이다.
사전 등록 설정: `config/extension_config.yaml` (2026-10-10T21:43:17+0900, sha256 6317bf5a…; 이후 수정 없음).

## A. 환경·입출력

1. **NAFNet 전용 별도 venv (`.venv_nafnet`)**, 출력은 무손실 PNG로 저장하고 픽셀 sha256으로 연결.
   이유: 기존 실험 환경(`.venv`)을 바꾸지 않기 위해. 후속 단계는 블러 입력을 base 코드로 다시 생성한 뒤,
   그 해시가 NAFNet manifest의 입력 해시와 같을 때만 출력을 사용한다(불일치 시 중단). 결과 확인 전.
2. **공식 arch 파일 3개를 수정 없이 로드**(stub `basicsr` 모듈 사용). 이유: 패키지 `__init__`이 lmdb·torchvision·학습
   코드를 끌어오기 때문. 클론 저장소는 `git status` 기준 수정 없음. 결과 확인 전.
3. **공식 테스트 옵션 그대로**(NAFNetLocal/TLC, 전체 영상, `grids: false`), fp32, 타일링·해상도 변경·fp16 없음.
   이유: 공식 절차 재현, 범위 밖 조작 금지. 결과 확인 전.
4. **회색조 → 3채널 동일 복제 입력, 출력은 공식 tensor2img 후 `cv2.COLOR_BGR2GRAY`**(base 전처리와 같은 변환).
   어떤 방법에도 색 정보를 주지 않는다. 결과 확인 전.
5. **TF32 off, cudnn deterministic, benchmark off.** 반복 실행 5장은 비트 단위로 같았고, CPU와 GPU 차이는
   3장에서 최대 1 grey level(차이 화소 ≤ 0.16 %). 결과 확인 전.
6. **VC++ 런타임은 venv 안에 app-local `msvcp140.dll`로 해결**(시스템 설치 없음).
7. **base 코드 hash 차이 확인**: 현재 `src/` hash(b5fa4afb…)가 main run 기록(e34adc88…)과 다른 이유는
   main run 이후 추가된 `src/repro_check.py` 하나뿐이다. 나머지 파일은 main run snapshot과 바이트 단위로 같고,
   repro_check.py를 빼면 hash가 e34adc88…로 일치한다. 파이프라인 동작 변화 없음.

## B. 단계 분리(Exp1)

8. **OpenCV SIFT `compute()`에 octave −1 sentinel 키포인트를 추가한 뒤 제거한다.**
   - 발견: `tools/probe_compute.py`, `tools/probe_sift_octave.py`(3쌍 × 8블러 × {B, Unsharp, Wiener} = 72장).
     ORB는 detect()+compute()가 detectAndCompute와 항상 비트 단위로 같았다. SIFT는 72장 중 8장에서 descriptor가 달랐고
     (원소 최대 차 13–16), 8장 모두 키포인트 최소 octave가 0 이상인 강한 블러 영상이었다.
   - 원인: SIFT `compute()`는 피라미드 시작 octave를 `min(0, 제공된 키포인트의 최소 octave)`로 정하지만,
     detectAndCompute는 항상 octave −1(2배 업샘플 기저)에서 시작한다.
   - 조치: `stagewise.describe()`가 octave −1 sentinel(class_id −2)을 붙여 같은 피라미드 규약을 쓰게 하고, 계산 후 제거한다.
     이렇게 하면 72/72가 비트 단위로 같았다. 본 실행에서는 모든 BB/RR/CLEAN 영상에서 비트 단위 일치를 다시 확인한다.
   - 이유: 구현 방식 차이(피라미드 기저)가 BR/RB 비교에 섞이지 않게 하기 위해. 결과 확인 전.
9. **키포인트 ID 추적**: compute() 전에 `class_id = 검출 인덱스`로 두고, 반환 후 제거·재정렬·속성 변화를
   행마다 기록한다(`n_removed_by_compute`, `reordered_by_compute`, `attrs_changed_by_compute`).
   상세 NPZ에는 각 단계의 `ids`(검출 인덱스)를 저장한다.
10. **BR/RB 행에는 영상 지표(PSNR 등)를 넣지 않는다.** 영상 지표는 영상의 속성이므로 `image_metrics.csv`
    (pair × blur × {B, unsharp, wiener, nafnet})에 따로 두고, 영상이 하나로 정해지는 BB·RR·CLEAN 행에만 붙인다.
11. **CLEAN/BB/RR은 재계산한 뒤 base 행과 같은지 검증한다(복사하지 않음).** base에는 영상·descriptor가 저장되어
    있지 않기 때문이다. 비교 대상은 CLEAN↔CLEAN default, BB↔A, RR_unsharp↔C, RR_wiener↔D이고, 시간 열을 뺀 모든 지표가
    정확히 같아야 한다(`check_consistency.py`).
12. **스모크 실행** `ext_smoke_main_20261010-223440`(1쌍)은 디버깅 전용이며 일관성 출력만 확인했다. 분석에 쓰지 않는다.
13. **불완전 실행** `ext_main_20261010-223520`: 호스트 절전으로 약 17.6시간 멈췄다가 세션 종료로 중단되었다.
    최종 CSV가 없어 아무 결과도 읽지 않았다(`INCOMPLETE.md`). 처음부터 새 run ID로 다시 실행했다.
    이후 모든 실행 스크립트는 `SetThreadExecutionState`로 프로세스 실행 중에만 절전을 막는다(시스템 설정 변경 없음).

## C. NAFNet 출력 발산 (2026-10-10 22:30–22:50, NAFNet 추론 직후 · 모든 매칭 실행 전)

14. **관찰**: NAFNet 출력 240장 중 23장은 raw 출력이 [0,1]을 크게 벗어나(예: −25~+24) 입력과 무관한 격자 모양
    포화 패턴이 되었다(출력 vs 입력 PSNR 3.7–17.0 dB; 나머지 217장은 20.2 dB 이상).
    해당 영상은 Clean 대상 10/24장, 블러 대상 13/192장(gaussian σ1.5 9, σ3.0 2, disk r3 2), 모션 블러 기준 영상 0/24장이다.
15. **구현 오류 여부 점검**(`tools/nafnet_diag*.py`): float64 CPU = fp32 GPU(같은 값), TLC 없는 NAFNet도 발산,
    학습 패치 크기 256×256 crop도 발산 → 정밀도·TLC·패딩·영상 크기는 원인이 아니다. 원본 **컬러** 영상에서도 같은 장면이
    발산한다(i_gonnenberg, i_castle) → 회색조 복제만의 문제가 아니다. 공식 demo 영상에서는 정상 범위 출력이 나온다.
    체크포인트는 공식 sha256과 일치하고 `params` 키만 있다(EMA 없음), strict load 통과.
    블록별 활성값 추적 결과, 발산은 level-3 인코더 28블록 구간에서 최대 |활성값|이 약 1e3에서 3e4로 커지며 일어난다.
    → 이 모델을 공식 절차대로 실행했을 때 일부 입력에서 생기는 실제 동작으로 판단한다. 원인(예: 학습 분포 밖의
    선명·약블러 입력에서의 활성값 폭주)은 **검증하지 않은 가설**이다.
16. **처리 방침**: 모델 교체·출력 수정·추정은 하지 않는다. 주분석은 사전 등록대로 **모든 셀을 포함**한다(실패 유지).
    민감도 분석만 추가한다. 이때 출력 전용 규칙 `chan_spread_max >= 255`로 발산 영상을 표시한다. 이 규칙은 R=G=B 입력인데
    어떤 화소에서 출력 채널이 0과 255로 정반대 극단에 있는 경우이며, Clean·GT·매칭 정보를 쓰지 않는다.
    채널 차 평균은 0.5–2 구간이 연속적이라 경계로 쓰지 않았다. 이 규칙은 NAFNet 출력만 보고 정했으며 **모든 매칭 결과를
    보기 전**에 확정했다(사후 결정이므로 여기에 기록). Clean 대상의 발산(10/24)은 "NAFNet은 선명한 입력에서 항등에 가깝게
    동작하지 않는다"는 진단 결과로 보고한다.

## D. 분석 정의

17. **추적 상태 분할**: 사전 등록된 세 상태(재검출 안 됨 / 재검출됐지만 올바른 매칭 아님 / 올바른 매칭)를 아래처럼 빠짐없이
    나눈다. correct_via_linked, correct_via_other(재검출된 위치가 아닌 다른 키포인트로 정답 매칭), lost_but_correct(3 px 안에서
    재검출되지 않았지만 다른 키포인트로 정답 매칭; 잔여 범주), not_redetected, redetected_no_match, redetected_wrong_match.
    "올바른 매칭" = 기준 키포인트 q_r의 ratio-test 매칭이 GT 오차 ≤ 3 px.
18. **재검출(연결) 판단은 detect() 키포인트 전체로 한다**(compute() 제거 전). 본 실행에서 compute() 제거는 0건으로
    확인했으므로 단계별 집합과 같다.
19. **변위 속성**: 위치가 같은 키포인트가 여럿이면(SIFT 다중 방향) 위치·크기·octave는 그 위치의 첫 키포인트를 쓰고,
    방향 차는 그 위치 키포인트들 중 최소 절대 차(±180° 감싸기)를 쓴다. 비교 대상 Clean 속성은 기준 집합의 Clean 키포인트 t_r.
20. **공간 분포의 유효 사각형**은 화소 중심 좌표 [m, W−1−m] × [m, H−1−m](`valid_grid`와 동일)로 한다. hull coverage는
    hull ∩ overlap 면적으로 계산해 1을 넘지 않게 한다.
21. **시간**: Unsharp/Wiener는 CPU 프로세스 내 측정, NAFNet은 GPU `t_total_ms`(메모리 내 uint8 → uint8, 모델 로드·warm-up·
    PNG I/O 제외). 단계 행의 검출·기술 시간은 detect()/compute()를 따로 잰 값이라 detectAndCompute 시간(base 비교용 열
    `runtime_detect_and_compute_ms`)과 다르다. 장치가 달라 공정한 속도 순위가 아니다. NAFNet 학습 비용은 알 수 없다.

## E. 사전 등록 당시의 결정 (backlog에서 옮김)

22. Exp2는 17 px 모션(base에서 단측 블러 효과가 가장 큰 조건; base Test 결과를 본 뒤 고름 → 탐색적). diff는 같은 길이의 직교 방향
    (기준 90°, 대상 0°). 24쌍 전부.
23. Exp3: 실제 블러 l17_a0. 제공 커널은 len13/len21(±4 px)과 +5°/+15°(끝점 횡변위 0.74/2.20 px), balance 0.005 고정.
    결과 확인 전에 정함. 탐색 없음.
24. 연결 반경 3 px, unique/ambiguous/lost, chance baseline ±20 px. 변위는 unique 연결에만, 3 px에서 절단(censored).
25. 고정 랜드마크 H: B·R 양쪽 검출에서 모두 unique로 연결된 기준 원소, 최소 8개, LS(method=0). 정체성과 개수를 고정한
    진단용이다.
26. Oracle-inlier H: GT-correct 매칭만으로 LS(method=0), 최소 8개. RANSAC/outlier 실패와 정답 대응점 부족을 구분하기 위한
    진단용이다.
27. 공간 분포는 기준 영상 좌표에서 overlap 다각형 기준(hull coverage, 4×4 점유율).
28. NAFNet 출력은 입력 픽셀 sha가 같을 때만 Exp1/Exp2에서 재사용.

## F. 분석 단계 결정 (2026-10-11)

29. **추정 실패율** = H를 얻지 못했거나 무효인 셀(`insufficient_matches`, `homography_estimation_failed`, `invalid_homography_*`)의
    비율. `projection_error_gt_3px/5px`는 "추정했지만 부정확"으로 실패율에 넣지 않는다(Success@3에는 실패로 반영).
30. **공간 분포 사례 선택 규칙**은 `analysis/scripts/a03_cases.py` docstring에 그림을 그리기 전에 고정했다:
    검출기별, 방법을 합친 셀에서 Δlog10 오차(RR−BB, 공통 유효 셀)의 10/50/90번째 백분위에 가장 가까운 셀 +
    gain 전이와 loss 전이 중 ΔGT-correct가 (하위) 중앙값인 셀. NAFNet 발산 셀은 제외하지 않고 표시만 한다.
31. **NAFNet 정렬 해석**: 크기는 240/240 일치하고, 패딩은 오른쪽·아래에만 붙여 `[:H,:W]`로 잘라내므로 원점 이동은 구조상 없다.
    전역 LS 이동 추정은 모션 블러 입력에서 중앙값 약 0.01 px였고, Gaussian/disk 입력에서는 영상별로 −1~+1.9 px의 겉보기
    이동이 나왔다(대조군 Wiener ≤ 0.006 px, B ≤ 0.056 px). 이 추정기는 순수 평행이동을 가정하므로, 비평행이동 복원 변화가
    섞이면 편향될 수 있다. 따라서 "정렬 오류"로 단정하지 않고 확인하지 못한 부분으로 보고한다.
32. **연관성 분석**(`a05_associations.py`)은 연구 질문 3에 답하기 위해 결과를 본 뒤 추가한 **탐색적** 분석이다.
    셀 수준 상관은 셀이 독립이 아니므로 기술적 수치로만 쓴다.
33. **Exp3의 BB 대비 비교**(불일치 커널 Wiener vs 무처리)는 사전 등록 대비(exact 대비)에 더한 기술적 비교다.
34. **그림**은 발표 슬라이드용 정적 PNG(light 배경)다. 색은 dataviz 기본 범주 팔레트 순서(슬롯 1–4)를 쓰고, 마커 모양을
    보조 부호화로 함께 쓴다. 이 환경에는 Node가 없어 팔레트 검증 스크립트를 실행하지 못했다. 그 대신 문서에 기록된
    검증 결과(light 모드 인접 쌍 통과)에 의존한다.
35. **ORB 추적 결과의 한계**: ORB는 키포인트가 촘촘해 3 px 연결의 대부분이 ambiguous다(기준 집합 중 unique 1.6–3.0 %).
    따라서 ORB의 변위 통계와 고정 랜드마크 H(분석 가능 셀 3.6–11.5 %)는 작고 선택된 부분집합에 근거하며, 이 제약을 보고서에
    명시한다. 규칙은 사전 등록대로 유지했다(사후 변경 없음).
