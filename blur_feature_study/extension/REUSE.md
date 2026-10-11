# REUSE — 확장 실험에서 재사용한 기존 결과·파일

기존 파일은 모두 **읽기 전용**으로 사용했고, `src/`, `config/`, `data/`, `results/`, `analysis/` 안의 어떤 파일도 수정하거나
덮어쓰지 않았다. 확장 실험의 코드·결과는 모두 `extension/` 안에 있다.

## 1. 그대로 재사용한 코드·설정 (import / read)

| 기존 파일 | 확장에서의 용도 |
|---|---|
| `config/experiment_config.yaml` | 전처리(회색조, 긴 변 960 INTER_AREA), 유효 여백 24 px, 블러 8조건, Unsharp α1.0 σ1.2, ratio 0.75, RANSAC 3 px/2000/0.995, 평가 설정 |
| `config/selected_params_validation_20261010-003108.yaml` | Wiener balance(8조건 모두 0.005), ORB/SIFT default 파라미터 |
| `data/dataset_manifest.csv`, `data/hpatches-sequences-release/` | Test 24쌍(12 시퀀스), 원본 영상, H (`src.dataset.load_pair`가 H_resized를 manifest와 대조) |
| `data/gt_quality_flags.csv` | GT 의심 플래그(v_cartooncity__1-5) |
| `src/io_utils.py`, `src/dataset.py`, `src/psf.py`, `src/imaging.py`, `src/features.py`, `src/metrics.py` | 블러 생성, Unsharp, Wiener, 검출·매칭·RANSAC, 모든 지표. 확장 코드는 이 함수들을 호출만 한다 |
| `src/pipeline.py` (`Runner._evaluate`) | 확장 `evaluate.py`가 같은 계산 순서·실패 사유를 재현(코드는 읽기만 함) |
| `tests/test_imaging.py::_shift` | NAFNet 정렬 점검의 전역 이동 추정기(`extension/src/nafnet_checks.py`에 그대로 복사, 출처 표기) |
| `analysis/scripts/common.py` | 통계 helper(`cluster_bootstrap_ci`, `paired_summary`, `mcnemar_exact`, `holm`, `spearman`)를 importlib로 불러 사용 |

## 2. 재사용한 기존 결과

| 기존 결과 | 용도 | 재사용 방식 |
|---|---|---|
| `results/main/main_20261010-003501/raw_results.csv` | 일관성 기준(CLEAN default, A, C, D 행) | **복사하지 않음.** 확장 실행이 CLEAN/BB/RR을 다시 계산하고, 시간 열을 뺀 31개 지표 열이 모두 정확히 같은지 검증(`ext_main_*/consistency.md`: 1200/1200행 일치). Exp3 exact 커널 48행도 base D와 일치(`ext_mismatch_*/consistency.json`) |
| `results/main/main_20261010-003501/snapshot/src/` | base 코드 hash 확인 | 현재 `src/`와 비교: 추가 파일 `src/repro_check.py` 외에는 바이트 단위로 동일 |
| `analysis/stats/clean_fail5_pairs.json`, `analysis/tables/T7_clean_baseline_per_pair.csv` | Clean 실패 쌍 확인용 참고 | 확장 분석은 새 CLEAN 행에서 Clean 실패 쌍을 다시 계산했다. @3 px 목록(ORB 6쌍, SIFT 1쌍)은 base `T7_clean_baseline_per_pair.csv`의 Success@3 = 0 목록과 같다(`clean_fail5_pairs.json`은 @5 px 기준: ORB 5쌍, SIFT 1쌍) |

base 실행은 영상과 descriptor를 저장하지 않았으므로(NPZ에 키포인트만 있음), B·R 영상과 특징은 base 함수로 결정적으로
다시 생성했다. 재생성 결과가 base와 같다는 것은 위 일관성 검사로 확인했다.

## 3. 확장 실행 안에서 재사용한 중간 산출물

| 산출물 | 생성 | 재사용 | 재사용 조건 |
|---|---|---|---|
| `results/ext_inputs_20261010-222114/images/*.png` | `prepare_inputs` (base 함수로 블러 생성) | NAFNet 입력 | 무손실 PNG + 픽셀 sha256 |
| `results/ext_nafnet_20261010-222216/images/*.png` | `nafnet_infer` (별도 venv, GPU) | Exp1(`run_main`), Exp2(`run_bothblur`), `nafnet_checks` | 각 스크립트가 블러 입력을 base 코드로 다시 만들고, 그 sha256이 NAFNet manifest의 입력 sha256과 같을 때만 사용. 출력 sha256과 크기도 확인(불일치 시 중단) |
| `results/ext_main_20261011-162418/details/*.npz` | `run_main` | `tracking` | 같은 run의 키포인트·매칭 기록. 추적 결과의 GT-correct 총수가 raw_results와 일치함을 확인(`ext_tracking_*/run_meta.json`) |

## 4. 사용하지 않은 실행

- `results/ext_smoke_main_20261010-223440`: 1쌍 스모크(디버깅 전용, 일관성 출력만 확인).
- `results/ext_main_20261010-223520`: 호스트 절전 후 세션 종료로 중단된 불완전 실행(`INCOMPLETE.md`). 아무 결과도 읽지 않음.
