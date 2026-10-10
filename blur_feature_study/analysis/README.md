# analysis/ — Main Experiment 심층 분석

실험 코드·원시 결과는 수정하지 않았다. 이 디렉터리는 `results/main/main_20261010-003501/raw_results.csv`(+NPZ, validation 결과)를 읽기만 한다.

## 산출물
| 경로 | 내용 |
|---|---|
| `analysis_report.md` | 전체 분석 과정·결과·해석 (상세) |
| `key_findings.md` | 핵심 발견과 근거 수치 |
| `research_assessment.md` | 학술적 의미, 한계, 논문 방향, 추가 실험 |
| `figures/*.png`, `figures/FIGURE_NOTES.md` | 탐색·검증용 그림과 그림별 설명 |
| `figures/visual_examples/` | 실제 영상·매칭 예시 (`07_visual_examples.py`) |
| `tables/T*.csv` | 집계표 (T1~T26) |
| `stats/S*.csv`, `stats/*.json`, `stats/log_0*.txt` | 통계 결과와 실행 로그 |
| `environment_analysis.txt` | 분석 환경 패키지 버전 |

## 재현
```powershell
# blur_feature_study/ 에서. 분석 전용 가상환경(analysis/.venv)은 실험용 .venv와 분리되어 있다.
$env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1     # 메모리 여유가 적은 호스트에서 필요
analysis\.venv\Scripts\python.exe analysis\scripts\01_consistency.py     # 설정·코드·CSV·NPZ 일치 점검
analysis\.venv\Scripts\python.exe analysis\scripts\02_descriptive.py     # 기술통계표 T1~T9
analysis\.venv\Scripts\python.exe analysis\scripts\03_paired_effects.py  # 대응 차이·CI·검정 S1, T10
analysis\.venv\Scripts\python.exe analysis\scripts\04_metric_relations.py # 지표 간 관계·모형 S2~S5, T11~T14
analysis\.venv\Scripts\python.exe analysis\scripts\05_mechanisms.py      # 원인 분석 S6~S9, T15~T26
analysis\.venv\Scripts\python.exe analysis\scripts\06_figures.py         # F01~F14
analysis\.venv\Scripts\python.exe analysis\scripts\07_visual_examples.py # 영상 예시
```
환경 생성: `python -m venv --without-pip analysis/.venv` 후 실험용 venv의 pip로 `pip --python analysis/.venv/Scripts/python.exe install -r requirements.txt matplotlib==3.10.1 "statsmodels>=0.14.5"`. (statsmodels 0.14.4는 scipy 1.17과 호환되지 않아 0.15.0을 설치했다.)

## 통계 처리 원칙 (`scripts/common.py`)
- 비교 단위는 **같은 영상 쌍·같은 블러·같은 검출기 셀 안의 대응 차이**다.
- 블러 8조건은 같은 쌍의 반복 측정이므로, 풀링 효과는 쌍 수준 평균(n=24)으로 먼저 줄인 뒤 계산한다.
- 같은 시퀀스의 두 쌍(target 2, 5)은 독립이 아니므로 CI는 **시퀀스(12개) 클러스터 부트스트랩**으로 구한다. Wilcoxon p는 쌍 수준(클러스터 무시)이므로 보조 지표다.
- 블러 조건별 검정은 조건 8개에 대해 Holm 보정을 적용했다.
- 영상 품질 지표(PSNR/SSIM/Edge F1)는 검출기와 무관하므로 영상 수준(576장)으로 중복을 제거한 뒤 분석한다.
- SIFT의 B는 A와 동일하므로 SIFT B−A 비교는 하지 않는다.
- H 추정 실패(오차 NaN)는 제외하지 않는다. 성공률에는 실패로, 오차 분포에서는 "미정의"로 표시하고, 순위 상관에서는 최악값으로 둔다.
