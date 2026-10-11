# 확장 실험: 복원 방법별 BB·BR·RB·RR 분해, 특징 추적, 양쪽 블러, 커널 불일치

base 연구(`../README.md`)의 Test 24쌍·8개 블러 조건을 그대로 쓰고, 비교군에 사전학습 NAFNet을 더해
복원의 적용 단계(검출 영상 / 기술자 영상)를 분리한다. 원 요청은 `TASK_SPEC.md`, 사전 등록 조건은
`config/extension_config.yaml`, 판단 기록은 `DECISIONS.md`, 재사용 기록은 `REUSE.md`, 결과 보고서는 `REPORT.md`에 있다.

## 실행 (모두 `blur_feature_study/`에서, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`)

```
.venv\Scripts\python.exe -m extension.src.prepare_inputs                       # -> results/ext_inputs_<ts>
.venv_nafnet\Scripts\python.exe -m extension.src.nafnet_infer --inputs extension/results/ext_inputs_<ts> --repeat-check 5 --cpu-check 3
.venv\Scripts\python.exe -m extension.src.nafnet_checks --nafnet-run ext_nafnet_<ts>
.venv\Scripts\python.exe -m extension.src.run_main --nafnet-run ext_nafnet_<ts>   # Exp1, 3888 rows
.venv\Scripts\python.exe -m extension.src.check_consistency --run ext_main_<ts>
.venv\Scripts\python.exe -m extension.src.tracking --run ext_main_<ts>
.venv\Scripts\python.exe -m extension.src.run_bothblur --nafnet-run ext_nafnet_<ts>  # Exp2, 960 rows
.venv\Scripts\python.exe -m extension.src.run_mismatch                          # Exp3, 240 rows
.venv\Scripts\python.exe -m pytest extension/tests -q -p no:cacheprovider       # 15 fast tests
# analysis (run ids in extension/analysis/runs.json)
analysis\.venv\Scripts\python.exe extension/analysis/scripts/a01_main.py
analysis\.venv\Scripts\python.exe extension/analysis/scripts/a02_tracking.py
analysis\.venv\Scripts\python.exe extension/analysis/scripts/a03_cases.py
analysis\.venv\Scripts\python.exe extension/analysis/scripts/a04_exp2_exp3.py
analysis\.venv\Scripts\python.exe extension/analysis/scripts/a05_associations.py
```

모든 실행은 새 run 디렉터리(`results/<prefix>_<YYYYMMDD-HHMMSS>/`)를 만들고 덮어쓰지 않는다. 각 run의 `snapshot/`에
실행 시점의 확장 코드와 설정이 복사된다(`run_meta.json`의 `ext_code_hash`는 실행이 끝난 시점의 `src/` 기준이므로,
실행 도중 다른 모듈이 추가된 경우 snapshot이 기준이다).

## 분석에 사용한 실행

| 단계 | run id | 행/영상 수 |
|---|---|---|
| NAFNet 입력 | `ext_inputs_20261010-222114` | 240장 (블러 대상 192, 블러 기준 24, Clean 대상 24) |
| NAFNet 추론 | `ext_nafnet_20261010-222216` | 240장, GPU |
| NAFNet 점검 | `ext_nafchk_20261011-163038` | 240장 |
| Exp1 | `ext_main_20261011-162418` | 3888행 (CLEAN 48 + BB 384 + BR/RB/RR 각 1152), 일관성 `consistency.md` |
| 추적 진단 | `ext_tracking_20261011-163146` | 상태 3888행, 연결 1536행, 고정 랜드마크 1152행 |
| Exp2 | `ext_bothblur_20261011-163226` | 960행 |
| Exp3 | `ext_mismatch_20261011-163426` | 240행, exact = base D 확인 |

## 환경

| 항목 | 값 |
|---|---|
| OS / HW | Windows 11 Pro 10.0.26300, Intel i5-14400F (16 logical), 16 GB RAM, NVIDIA RTX 4060 8 GB (driver 610.88) |
| `.venv` (모든 실험) | Python 3.11.9, numpy 2.4.6, opencv-contrib-python 5.0.0.93, scipy 1.17.1, scikit-image 0.26.0, pandas 3.0.6 |
| `.venv_nafnet` (NAFNet만) | Python 3.11.9, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900), numpy 2.4.6, opencv-contrib-python 5.0.0.93, PyYAML 6.0.3; app-local `msvcp140.dll` (DECISIONS A6) |
| `analysis/.venv` (분석) | Python 3.11.9, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.10.1, statsmodels 0.15.0, OpenCV 5.0.0, scikit-image 0.26.0 |
| NAFNet | 공식 저장소 megvii-research/NAFNet @ 2b4af71ebe098a92a75910c233a3965a3e93ede4, 옵션 `options/test/GoPro/NAFNet-width32.yml` (NAFNetLocal, width 32, enc [1,1,1,28], mid 1, dec [1,1,1,1]); 가중치 `NAFNet-GoPro-width32.pth` (공식 README gdrive id 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj, 68,671,121 B, sha256 19394e61…1d5a5c), key `params`, strict load, 17,111,907 parameters. 재학습·미세조정 없음. **사전학습 비용은 알 수 없다.** |
| 비공개 파일 | `third_party/`(저장소·가중치), `results/*/images/`, `results/*/details/`는 `.gitignore` |

## 시간 측정 범위

- Unsharp / Wiener: CPU, 같은 프로세스 안에서 `perf_counter`로 측정(OpenCV 스레드 16, BLAS 1 스레드).
- NAFNet: GPU(RTX 4060), `t_total_ms` = 메모리 안의 uint8 회색조 → 텐서 → GPU → forward → CPU → uint8 회색조
  (`torch.cuda.synchronize` 양끝). 모델 로드(0.94 s), warm-up(0.45 s), PNG 입출력 제외. 참고용 CPU 1스레드
  3장(385×640): 장당 약 2.1–2.3 s.
- 검출·기술·매칭·H 시간은 단계 행에서 detect()/compute()를 따로 잰 값이다. base와 비교할 수 있는 값은
  `runtime_detect_and_compute_ms`(BB/RR/CLEAN)이다.
- 장치가 달라(CPU vs GPU) 실행 시간은 방법 간 공정한 속도 순위가 아니다.
