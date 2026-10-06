# 실행과 재현

[README](../README.md) · [주요 실험](experiments.md) · [결과와 평가 한계](results.md)

## 실행 환경

당시 학습·추론 장비는 MacBook Pro, Apple M5, 메모리 24GB였습니다. 운영체제는 macOS 26.6.2 arm64, Python은 3.12.13, PyTorch는 2.14.0이며 학습에는 MPS FP32와 `PYTORCH_ENABLE_MPS_FALLBACK=0`을 사용했습니다. 화면 크기와 CPU·GPU 코어 수는 기록에 없습니다.

세 학습 방법×3개 fold의 **9회 학습과 지정 평가**에 걸린 시간은 5,435.781초(약 90.6분)입니다. 최종 앙상블의 6개 모델만 따로 학습한 시간이 아니며 데이터 준비·합성 검사·보고서 작성은 포함하지 않습니다.

확인한 패키지 버전은 Python 3.12.13, PyTorch 2.14.0, NumPy 2.5.3, Shapely 2.1.2, scikit-image 0.26.0입니다. 고정 버전은 [pyproject.toml](../pyproject.toml)을 따릅니다. 집계 그림에는 선택 의존성인 Matplotlib 3.11.1을 사용했습니다.

| 확인 날짜 | CPU 합성 검사 | MPS 합성 검사 |
|---|---|---|
| 2026-10-05 | 61개 통과 | 별도 실행 3개 통과 |
| 2026-10-07 | 61개 통과 | 미실행 — 해당 실행 환경에서 MPS 사용 불가 |

기존 의존성이 있는 환경에서 오프라인 패키지 빌드·임시 경로 설치·설치된 모듈 import를 확인했습니다. 2026-10-07에는 macOS 27.0.1 arm64·Python 3.12.13의 빈 가상환경에서 전체 의존성을 내려받아 기본 설치(`pip install -e .`)와 그림용 선택 설치(`pip install -e '.[figures]'`)를 모두 마쳤습니다. `pip check`에서 의존성 충돌이 없었고, 같은 환경에서 CPU 합성 검사 61개도 통과했습니다.

원자료 준비·전체 재학습·실제 영상과 원 가중치의 출력 일치·CUDA·다른 운영체제는 이 저장소의 코드로 검증하지 않았습니다. 플랫폼 평가 자료가 없어 대회 점수도 다시 계산하지 못했습니다. 원 제출 기록에는 실제 1,691키와 검증 343키의 출력 검사가 남아 있으나 위 합성 검사와는 별개의 기록입니다. MPS 합성 통과를 실제 자료 전체 학습이나 최종 폴리곤의 장치 간 일치로 확대하지 않습니다.

## 설치와 합성 검사

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests/integration -v
```

검사에서 생성하는 영상·정답·가중치는 모두 합성입니다. 미라벨이 감독에서 제외되는지 확인하고 작은 양성과 무효 GT의 처리도 검사합니다. 회전·반사 후 원래 방향 복원과 확률 평균, 면적 비율 정답·사이트별 Dice 가중치, split 누수, 체크포인트 선택·복구를 검사합니다. 역할별 앙상블 조립과 별도 프로세스 CLI도 확인합니다. 임시 파일은 검사가 끝나면 삭제합니다.

MPS가 있는 환경에서 별도로 확인하려면 다음과 같이 실행합니다. 2026-10-05 이 명령의 합성 MPS 검사 3개를 통과했고, 2026-10-07에는 MPS를 사용할 수 없어 실행하지 않았습니다. 실제 데이터·원 가중치의 MPS 검증과는 구분합니다.

```bash
LANDSLIDE_TEST_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src \
  python -m unittest tests.integration.test_unet_runtime.UnetMPS tests.integration.test_tta.TTAMPS -v
```

## 데이터 준비

학습 원자료는 `data/train/<location>/bands/`, `sites.json`, `labels/YYYYMMDD.json` 구조를 유지합니다. labels의 행은 `site_id`, `visible`, `polygon`을 포함합니다. 미라벨을 음성으로 채우지 않으며 작은 visible도 Presence 양성입니다. 원자료는 수정하지 않습니다. 비어 있지 않은 무효 양성 GT 3건에는 원 계약을 적용합니다. Presence 감독을 유지하고 segmentation에서 제외합니다. 빈 양성이나 예상 밖 감독 결함은 자동으로 보정하지 않습니다.

추론 입력은 지역 폴더별 `bands/YYYYMMDD.npy`와 `sites.json`입니다. 배열은 H×W×4의 R·G·B·NIR 순서이며 한 행의 예측 단위는 `(location_id, date, site_id)`입니다. `sites.json`의 `site_id`와 `prior_polygon`을 사용합니다. prior는 관심 사이트의 범위를 나타내는 입력이며 날짜별 정답 polygon과 구분합니다. 추론은 정답 `labels/`나 미리보기 `images/`를 읽지 않습니다.

```text
data/inference/
└── synthetic_region/
    ├── sites.json
    └── bands/
        └── 20200101.npy
```

`sites.json` 형식의 합성 예시입니다. 아래 좌표는 실제 지역에서 가져오지 않았습니다.

```json
[{"site_id": "001", "prior_polygon": [[[0,0],[24,0],[24,24],[0,24],[0,0]]]}]
```

데이터 취득과 이용 조건은 [README](../README.md#한계와-이용-조건)를 따릅니다. 아래 명령은 이미 확보한 자료의 경로를 지정하며 데이터를 자동으로 내려받지 않습니다.

## 학습

[configs/splits](../configs/splits)와 [configs/training](../configs/training)에 배정과 수치 설정을 보관했습니다. 원 학습 12개 지역의 각 outer fold는 평가 4개 지역을 제외하고 나머지를 학습 6개·내부 선택 2개로 나눕니다. 원 holdout 3개 지역은 이 학습·선택·outer 평가에서 모두 제외합니다. 연결 그룹과 중복 영상의 분할 교차를 검사합니다. 부분적인 지리 중첩 전체를 해소했다고 확인한 것은 아닙니다.

| fold | 내부 선택 지역 |
|---|---|
| 1 | `ls_scene_01_w03`, `ls_scene_01_w06` |
| 2 | `ls_scene_01_w08`, `ls_scene_01_w10` |
| 3 | `ls_scene_01_w05`, `ls_scene_01_w08` |

다음 준비 명령도 **실제 학습 자료를 읽으므로** 자료 사용 권한을 확보한 환경에서 실행합니다. 분할·감독·학습 전용 통계와 coverage 정답을 준비하고 검사하지만 optimizer update는 수행하지 않습니다.

```bash
python scripts/run_ab_experiment.py \
  --data-root data/train --config-dir configs --output runs/training-check --device cpu
```

아래 명령은 기록된 MPS 조건을 유지합니다. CPU 학습은 `--device cpu`로 실행할 수 있습니다. 장치 변경은 학습 예산·seed·분할·선택 정책을 바꾸지 않지만 원 MPS 결과와의 수치 일치를 보장하지 않습니다. CPU로 준비한 폴더와 실행 설정이 다르므로 새 출력 폴더를 사용합니다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_ab_experiment.py \
  --data-root data/train --config-dir configs --output runs/training-mps --device mps --execute
```

fold 1→2→3, 각 fold에서 baseline→A→B 순서로 9개 모델을 학습합니다. 모든 학습이 끝난 뒤 outer 자료를 평가합니다. 각 실행은 5,000 update, batch 32, Adam 0.001, 고정 학습률, FP32이며 seed는 42입니다. 표본 순열과 회전·반사 증강 추첨은 각각 독립된 `numpy.default_rng(42)`를 사용합니다. 순열 끝의 작은 배치는 버리거나 다음 순열로 채우지 않습니다. 부분 학습 실행이 있으면 자동으로 다시 시작하지 않으며 완료한 학습은 보존합니다.

이 9회 비교 학습에서 얻은 기준 모델 3개를 최종 Presence에, 면적 비율 모델 3개를 최종 분할에 사용합니다. 사이트별 Dice 가중치 모델 3개는 비교용이며 최종 구성에서 제외합니다.

## 재학습 가중치 조립

아래 경로는 앞의 학습 명령으로 생성됩니다. 각 역할에 fold 1→2→3의 `best_d4.pt`를 전달합니다. `--retrained`는 원 S030의 update 수를 강제하지 않고 각 실행의 `validation_history.json`에서 참고종합이 처음 최고가 된 체크포인트 선택과 저장된 메타데이터를 대조합니다. fold 순서, 모델 역할, 두 역할의 입력·정규화 계약은 계속 확인합니다.

```bash
python scripts/prepare_coverage_submission.py --retrained \
  --segmentation runs/training-mps/fold1/A/best_d4.pt runs/training-mps/fold2/A/best_d4.pt runs/training-mps/fold3/A/best_d4.pt \
  --presence runs/training-mps/fold1/baseline/best_d4.pt runs/training-mps/fold2/baseline/best_d4.pt runs/training-mps/fold3/baseline/best_d4.pt \
  --output weights/retrained.pt
```

조립기는 가중치와 같은 이름의 JSON을 만들고 새 학습에서 사용한 fold별 통계 및 원래의 회전·반사 평균·Presence 0.2·등위선 0.4 정책을 저장합니다. 기존 가중치나 JSON은 덮어쓰지 않습니다. 결과는 `retrained_role_ensemble`로 표시하며 기존 S030의 플랫폼 성적을 이 모델의 성적으로 붙이지 않습니다. `--retrained`를 생략한 원 S030 조립 검사는 그대로 유지합니다.

## 추론

앞에서 조립한 재학습 가중치에는 같은 이름으로 생성된 추론 JSON을 사용합니다. 원 가중치용 고정 설정으로 바꾸면 재학습한 fold별 정규화 통계가 달라질 수 있습니다.

```bash
python scripts/run_inference.py \
  --input data/inference --checkpoint weights/retrained.pt \
  --config weights/retrained.json --output outputs/retrained.csv --device cpu
```

`--input`과 `--checkpoint`는 읽기용이며 출력은 입력 폴더와 체크포인트 밖에 둡니다. CLI는 기존 `run_inference`에 인자를 전달하고 CPU 스레드를 2개로 설정합니다. `--device mps`와 `--device cuda`도 전달할 수 있지만 공개본의 실제 가중치 실행과 장치 간 출력 일치는 미검증입니다.

입력은 [데이터 준비](#데이터-준비)의 지역·날짜·사이트 구조를 따릅니다. 출력 CSV는 모든 사이트·날짜의 행을 유지하며 **산사태 없음**은 `polygons` 칸을 비워 표시합니다. 좌표와 폴리곤 중첩 형식은 [폴리곤 변환과 출력](#폴리곤-변환과-출력)에 정리했습니다.

## 기존 최종 가중치로 추론

저장소에는 학습된 가중치가 없습니다. 사용 가능한 원 최종 가중치가 있다면 아래 절차를 따릅니다. 다운로드 주소와 재배포 조건은 미확인이며 제공을 약속하지 않습니다. [이용 조건](../README.md#한계와-이용-조건)을 확인한 가중치만 사용합니다.

최종 가중치의 분할 모델은 면적 비율 학습 모델 3개, Presence 모델은 **기준·면적 비율·사이트 가중치를 비교한 9회 학습에서 얻은 기준 모델 3개**입니다. 과거 실험에도 같은 `E086` 표기가 있으나 그때의 파일을 최종 Presence 가중치로 대신하면 안 됩니다.

| 역할 | 계열 | fold 1 | fold 2 | fold 3 |
|---|---|---:|---:|---:|
| segmentation | E090/A `best_d4.pt` | 4750 | 4500 | 1500 |
| Presence | 9회 비교 학습의 기준 E086 `best_d4.pt` | 2000 | 1750 | 2500 |

표의 수치는 원 체크포인트의 선택 update입니다. 재학습한 모델의 선택 update가 달라졌다면 앞의 `--retrained` 조립 절차를 사용하며 원 최종 모델의 성적으로 표시하지 않습니다.

여섯 원 파일을 다음처럼 배치하고 각 역할의 인자를 fold 1→2→3 순서로 전달합니다. 통합 가중치가 있다면 `weights/model.pt`에 바로 둘 수 있습니다.

```text
weights/
├── fold1/
│   ├── A/best_d4.pt
│   └── baseline/best_d4.pt
├── fold2/
│   ├── A/best_d4.pt
│   └── baseline/best_d4.pt
└── fold3/
    ├── A/best_d4.pt
    └── baseline/best_d4.pt
```

```bash
python scripts/prepare_coverage_submission.py \
  --segmentation weights/fold1/A/best_d4.pt weights/fold2/A/best_d4.pt weights/fold3/A/best_d4.pt \
  --presence weights/fold1/baseline/best_d4.pt weights/fold2/baseline/best_d4.pt weights/fold3/baseline/best_d4.pt \
  --output weights/model.pt
```

출력 파일이 있으면 덮어쓰지 않습니다. 조립은 추가 학습을 하지 않으며 원 S030의 ID·선택 update·정규화 통계와 다르면 중단합니다. 새로 학습한 모델의 선택 update가 달라졌다면 그것을 원 S030으로 표시하거나 검사를 우회하지 않습니다.

조립기는 실험 ID·선택 update·fold별 입력·정규화 설정을 대조합니다. 메타데이터만으로 파일의 실제 출처를 증명할 수는 없으므로 원 자산의 출처를 함께 확인해야 합니다.

통합 가중치에는 원 설정을 사용해 추론합니다.

```bash
python scripts/run_inference.py \
  --input data/inference \
  --checkpoint weights/model.pt \
  --config configs/inference_s030.json \
  --output outputs/predictions.csv \
  --device cpu
```

## 구현 세부 사항

### 전처리

RGBNIR을 10,000으로 나눠 반사율로 변환합니다. 별도의 offset은 더하지 않습니다. prior 중심의 96×96 crop을 `shift_inside` 정책으로 영상 안쪽에 옮기고 영상보다 큰 부분은 padding합니다. prior polygon을 래스터화한 채널과 `NDVI=(NIR-R)/(NIR+R)`를 더해 6채널 입력을 만듭니다.

공식 유효성, 수치 유효성, 원영상 내부 여부는 각각 유지합니다. 사용할 수 없는 화소는 정규화 뒤에도 0으로 두며 최종 polygon에서도 제외합니다. NDVI는 반사율 FP32를 float64로 변환해 계산하고 FP32로 저장합니다. 분모 epsilon은 `1e-8`이며 사용할 화소의 잘못된 지수 값은 자동 보정하지 않고 오류로 처리합니다.

RGBNIR의 z-score는 fold별 학습 6개 지역의 감독 표본에 있는 유효 화소에서만 적합합니다. 겹치는 ROI는 표본마다 계산에 포함됩니다. prior와 NDVI에는 이 RGBNIR 정규화를 적용하지 않습니다. 원 최종 가중치의 추론에는 [고정된 세 통계](../configs/inference_s030.json)를 각각 적용합니다. 재학습 모델의 추론에는 조립기가 만든 JSON에 저장된 fold별 통계를 사용합니다.

### 모델 구조와 초기화

SmallUNet의 채널 폭은 `[16,32,64,128]`이며 6채널 모델의 파라미터 수는 487,858개입니다. segmentation logit과 Presence logit을 함께 출력합니다. 먼저 5채널 모델을 초기화한 뒤 첫 합성곱의 입력 채널을 6개로 늘리고 추가한 채널의 가중치를 0으로 설정합니다. 이 순서는 학습 난수 상태와 초기 가중치에 영향을 주므로 재학습에도 그대로 적용합니다.

### 손실과 체크포인트 선택

기준 모델(E086/baseline)은 유효 segmentation 화소의 binary BCE와 유효한 양성 ROI의 binary Dice를 각각 0.5씩 더하고 Presence BCE를 더합니다. 배치에 Dice 참여 양성이 없으면 segmentation BCE의 계수는 1입니다.

면적 비율 모델(E090/A)은 BCE target만 FP32 `coverage_target`으로 바꿉니다. `coverage_target`은 원 GT와 unit pixel의 정확한 교집합 면적을 나타냅니다. binary Dice, Presence, 양성 판정은 유지합니다. 사이트별 Dice 가중치 모델(E091/B)은 학습의 Dice 참여 양성 ROI를 사이트별로 세어 `w=N_positive/(K_positive*n_site)`를 고정하고 배치 양성의 `mean(w_i*Dice_i)`를 사용합니다. 가중치 합으로 다시 나누지 않으며 A와 B를 결합하지 않습니다.

매 250 update마다 일반 검증과 회전·반사 평균 검증을 수행합니다. 일반 검증 손실은 학습 손실 종류에 관계없이 비가중 binary segmentation BCE + Presence BCE이며 유효 화소·표본 수로 집계합니다.

| 파일 | 선택 규칙 |
|---|---|
| `best.pt` | 일반 검증 BCE가 처음 최저가 된 시점 |
| `best_d4.pt` | 내부 선택의 회전·반사 평균·Presence 0.2·contour 0.4 참고종합이 처음 최고가 된 시점 |
| `last.pt` | 마지막 검증 시점 |

회전·반사 평균의 체크포인트 선택은 `0.6×Presence+0.4×Shape 진단`을 사용합니다. 동률이면 앞선 체크포인트를 유지합니다. outer 결과는 체크포인트 선택에 사용하지 않습니다. `best_d4`가 주 비교이고 `last`가 보조 비교입니다. 공개 설정의 `best_d4.pt`는 기록된 선택 규칙이며 같은 장치에서도 재학습 결과의 비트 단위 일치까지 보장하지 않습니다.

### 회전·반사 평균과 역할별 앙상블

각 fold에서 네 회전과 좌우 반전의 조합인 8개 변환을 적용합니다. segmentation에는 sigmoid를 적용하고 역변환한 확률을 평균합니다. Presence도 sigmoid 확률을 평균합니다. 그 뒤 segmentation은 E090/A 세 fold, Presence는 E086 세 fold를 각각 1/3씩 평균합니다.

`CoverageFoldEnsemble`의 각 member는 `BaselineLogitBlend(candidate_only=True)`입니다. 이 경로는 A의 segmentation logit과 E086의 Presence logit을 꺼냅니다. 클래스에 남아 있는 일반적인 절반 logit 혼합 경로는 S030에서 사용하지 않습니다. 학습과 역할 결합 후에 정규화 재적합, 추가 학습, 비중 탐색을 하지 않았습니다.

### 폴리곤 변환과 출력

평균 Presence 확률이 0.2 미만이면 **산사태 없음**으로 판정해 폴리곤을 출력하지 않습니다. 통과하면 segmentation 확률의 0.4 등위선으로 경계를 추출합니다. 무효 화소를 0으로 놓고 바깥에 배경을 padding한 뒤 `find_contours(..., fully_connected="low")`를 사용합니다. 화소 중심 좌표와 ROI 원점을 보정하고 hole·분리된 부분을 구성해 원영상의 유효 영역으로 제한합니다. 면적 기준으로 작은 예측을 제거하거나 잘못된 도형을 임의로 수선하지 않습니다. 퇴화한 0면적 경계는 별도로 셉니다.

[추론 설정](../configs/inference_s030.json)의 상위 `presence_threshold=0.5`, `mask_threshold=0.5`는 저장된 모델 설정을 확인하는 값입니다. S030의 실제 후처리는 중첩된 `postprocessing`의 `presence_threshold=0.2`, `mask_threshold=0.4`, `method=probability_contour`를 사용합니다. 이 두 설정을 합쳐 바꾸면 체크포인트 계약이 달라집니다.

출력 CSV 열은 `location_id,date,site_id,polygons`입니다. `polygons`는 GeoJSON MultiPolygon과 같은 `[polygon][ring][point][x,y]` 중첩의 JSON 문자열이며 첫 ring은 외곽선이고 나머지는 hole입니다. 좌표는 원영상의 연속 픽셀 좌표입니다. 왼쪽 위가 원점이며 x는 오른쪽, y는 아래쪽으로 증가합니다. 산사태 없음은 `polygons` 칸을 비워 기록하며 모든 사이트·날짜의 행을 유지합니다. 파일 쓰기에 실패하면 기존의 완성된 CSV를 부분 출력으로 덮어쓰지 않습니다.

재학습 모델에도 상위 체크포인트 확인 값과 실제 후처리 값의 구분은 유지합니다. 다만 추론 설정은 고정된 원 설정 대신 조립기가 생성한 JSON을 사용합니다.

## 코드·설정 표기 대응

README는 모델의 역할과 학습 방법으로 설명합니다. 코드·설정·기존 그림에는 아래 식별자가 남아 있으므로 파일을 찾거나 설정·그림과 대조할 때 이 표를 사용합니다.

| README의 설명 | 기록·코드의 표기 | 구성 또는 역할 |
|---|---|---|
| 최종 모델 | `S030` | 면적 비율 모델의 분할 3개와 기준 모델의 존재 판정 3개를 결합 |
| 역할 분리 전 구성 | `S029` | 면적 비율 모델 3개에서 분할과 존재 판정을 모두 사용 |
| 이진 정답으로 학습한 기준 모델 | `E086`, `baseline` | 9회 비교 학습에서 얻은 기준 모델 3개. 최종 구성에서는 존재 판정에 사용 |
| 면적 비율을 학습한 모델 | `E090`, `A` | segmentation BCE에 픽셀을 덮는 정답 면적 비율을 사용. 최종 구성에서는 분할에 사용 |
| 사이트별 Dice 가중치를 적용한 모델 | `E091`, `B` | 기준 모델의 Dice 손실에 사이트별 가중치를 적용한 비교. 최종 구성에는 미사용 |
| 학습 지역을 늘린 비교 모델 | `E121` | 학습 지역을 조합당 6개에서 8개로 늘린 후속 앙상블. 비교 결과만 포함 |

`D4`는 코드·파일명에서 회전·반사 8개 변환을 가리키는 표기입니다. 이 저장소의 실제 평균 방식은 [회전·반사 평균과 역할별 앙상블](#회전반사-평균과-역할별-앙상블)을 따릅니다.

초기 U-Net·입력·손실·후처리와 체크포인트 선택 비교의 실행 명령은 [주요 실험](experiments.md)에 있습니다.
