# 주요 실험

[README](../README.md) · [최종 구성](../README.md#최종-해법) · [결과와 평가 한계](results.md) · [학습·추론](reproduction.md)

최종 해법에 영향을 준 입력·손실·후처리·체크포인트 선택과 역할별 앙상블 비교를 정리했다. 수치는 당시 실험 기록이며, 비교마다 평가 자료와 조건이 다르다. [평가 산식과 범위](results.md#평가-단위와-산식)를 함께 확인한다.

## U-Net 기준선·입력·손실

공통 조건은 seed 42, 원 지역·연결 그룹 holdout, SmallUNet, Adam 0.001, batch 최대 32, 5,000 update, 무작위 회전·반사 학습 증강이다. 250 update마다 일반 검증의 segmentation BCE와 Presence BCE 합을 측정해 처음 최솟값인 `best.pt`를 선택한다. Dice를 학습 손실에 추가해도 선택식에는 Dice를 넣지 않는다.

| 설정 | 이전 단계에서 바꾼 내용 | 유지·관측한 내용 |
|---|---|---|
| [E001](../configs/experiments/E001.json) | 첫 기준선: RGBNIR+prior, 96px 중심 ROI, BCE | 영상 밖은 padding. 기준 실행과 첫 제출은 서로 다름 |
| [E002](../configs/experiments/E002.json) | ROI 원점을 영상 안쪽으로 이동 | 픽셀 해상도·크기·손실 유지 |
| [E005](../configs/experiments/E005.json) | segmentation BCE와 binary Dice를 결합 | BCE 기준보다 누락 감소, 오탐 증가 |
| [E008](../configs/experiments/E008.json) | NDVI 채널 추가 | 표준화 전 반사율에서 계산. 추가 채널의 초기 가중치는 0 |
| [E015](../configs/experiments/E015.json) | RGBNIR에 학습 전용 z-score 적용 | 감독 train ROI의 유효 화소만 적합. prior·NDVI는 그대로 |

E002와 E005의 원 `best.pt`, 일반 추론, 임계값 0.5/0.5 결과는 다음과 같다. 같은 holdout 343키 중 감독 319건, 양성 120·음성 199건을 평가했다. Shape 진단은 89날짜·9사이트이며 무효 GT 2건 때문에 전체 Shape·Total은 NA다.

| 조건 | Presence | Shape 진단 | 참고종합(약) | E002 대비(약) | FP / FN |
|---|---:|---:|---:|---:|---:|
| E002 BCE | 0.698719 | 0.201707 | 0.499914 | — | 0 / 74 |
| E005 BCE+Dice | 0.900759 | 0.476237 | 0.730950 | +0.231036 | 18 / 12 |

Presence는 최종 유효 polygon의 존재로 계산한 양성·음성 F1 평균이다. Shape는 유효한 면적 10px² 이상 양성의 연속 polygon IoU를 날짜→사이트 순서로 평균한다. 참고종합은 `0.6×Presence+0.4×Shape 진단`이며 위 값은 표시된 성분값에서 계산한 근삿값이다. 작은 양성도 Presence에는 포함한다. 정확한 [산식과 NA 조건](results.md#평가-단위와-산식)을 함께 적용한다.

BCE+Dice에서는 누락이 74→12건으로 줄고 오탐이 0→18건으로 늘었다. 참고종합이 높아져 결합 손실을 이후 기준으로 유지했다. NDVI와 학습 전용 정규화도 최종 입력에 남았지만 이 표에서 각 변경의 단독 개선 폭을 분리해 주장하지 않는다.

데이터 사용 권한을 확보한 환경에서 다음과 같이 실행한다. `--execute`를 빼면 실제 자료를 읽고 분할·입력·학습 통계만 준비한다. CPU도 지원하며 당시 장치는 MPS였다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_comparison.py \
  --data-root data/train --config configs/experiments/E015.json \
  --output runs/E015 --device mps --execute
```

다른 행은 `--config`와 `--output`을 바꿔 각각 실행한다. 과거 실행 폴더나 기존 가중치가 필요하지 않다. 원 동결 split과 학습 정의를 재사용하며 결과의 비트 단위 일치를 보장하지 않는다. 완료한 학습은 재사용하고 부분 학습은 자동 재시작하지 않는다.

## 같은 가중치의 후처리 비교

위 실행은 원 선택 규칙으로 저장한 `training/best.pt`를 사용해 회전·반사 8개 변환의 평균 확률을 구하고, Presence 0.2·분할 0.4에서 픽셀 사각형 합집합(`baseline`)과 확률 등위선(`probability_contour`)을 비교한다. 두 방법은 같은 가중치·확률·평가 표본을 사용한다. `baseline_d4.json`, `probability_contour_d4.json`에 지표와 예측이 저장된다.

현재 실행 코드는 초기 학습 설정에 **같은 후처리 진단 조건**을 적용한다. 모든 초기 실험에서 당시부터 0.2/0.4·등위선을 썼다는 뜻이 아니다. 원 일반 추론 0.5/0.5 검증 결과는 `training/fit_summary.json`과 `validation_history.json`에 따로 남는다. 후처리 결과로 `best.pt`를 다시 선택하지 않는다.

당시 같은 E015 가중치·회전·반사 평균·0.2/0.4에서 등위선을 적용한 S011의 Public은 0.7179618187, 픽셀 합집합 S010은 0.7078250947로 차이는 +0.0101367240이었다. 이는 저장된 플랫폼 Total이며 성분별 점수는 미확인이다. 다른 가중치에서도 같은 개선을 보장하지 않는다.

새 학습 가중치와 그 통계로 추론하려면 생성된 설정을 함께 전달한다.

```bash
python scripts/run_inference.py \
  --input data/inference --checkpoint runs/E015/training/best.pt \
  --config runs/E015/inference.json --output outputs/E015.csv --device cpu
```

## 회전·반사 평균을 적용한 체크포인트 선택

원 12개 개발 지역을 fold마다 학습 6·선택 2·outer 평가 4개 지역으로 나눴다. 일반 BCE 최소, 회전·반사 평균의 참고종합 최대, 마지막 update를 같은 학습 궤적에서 비교했다. 선택에는 내부 2개 지역만 사용했고 outer 결과는 사용하지 않았다. 동률이면 먼저 기록한 시점을 유지했다.

| 정책 | Presence | Shape 진단 | 참고종합 | BCE 선택 대비 | FP / FN |
|---|---:|---:|---:|---:|---:|
| 일반 BCE 최소 | 0.568959 | 0.486094 | 0.535813 | — | 457 / 90 |
| 마지막 5,000 update | 0.779262 | 0.519591 | 0.675394 | +0.139581 | 111 / 158 |
| 회전·반사 평균의 참고종합 최대 | 0.810612 | 0.528286 | 0.697682 | +0.161869 | 114 / 120 |

평가 대상은 pooled outer 1,348키 중 감독 1,282건이다. Shape 진단은 381날짜·33사이트이며 무효 GT 1건으로 전체 Shape·Total은 NA다. 위와 같은 산식을 사용한다. fold 점수를 단순 평균하지 않고 표본을 합쳐 다시 계산했다. 회전·반사 평균의 체크포인트 선택은 BCE보다 오탐을 줄였지만 누락은 늘렸다. 마지막 모델보다 Shape가 개선된 fold는 하나였으므로 전 지역의 경계 개선으로 해석하지 않는다.

`run_ab_experiment.py`는 `best.pt`, `best_d4.pt`, `last.pt`를 모두 저장한다. 기본 outer 평가는 `best_d4`와 `last`다. 일반 BCE 선택도 비교하려면 새 실행 폴더에 `--include-bce-control`을 지정한다. optimizer update 수와 학습 순서는 그대로이고 추가되는 것은 저장된 `best.pt`의 outer 평가다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/run_ab_experiment.py \
  --data-root data/train --config-dir configs --output runs/ab-with-bce \
  --device mps --execute --include-bce-control
```

## 면적 비율 학습과 역할별 앙상블

기준 모델(E086/baseline), 면적 비율 모델(E090/A), 사이트별 Dice 가중치 모델(E091/B)은 같은 fold·seed·표본 순서·회전·반사 증강 추첨을 사용한다. A는 segmentation BCE 정답만 픽셀의 정확한 GT 면적 비율로 바꾸고 binary Dice·Presence를 유지한다. B는 사이트별 양성 수에 따른 Dice 가중치를 비교한다. A와 B를 동시에 적용하지 않는다.

같은 outer 평가에서 A의 참고종합은 baseline 대비 +0.010020, B는 −0.005023이었다. A의 Shape 이득 중 약 69.6%는 평가 날짜가 하나인 사이트의 새 검출에서 나왔다. 작은 양성 FN은 총 41건으로 같았지만 회복 7건·신규 7건이 있었다. [전체 성분·오류·분모](results.md#구성-모델의-outer-oof-비교)를 함께 확인한다.

최종 앙상블(S030)은 이 9회 비교 학습에서 얻은 면적 비율 모델의 분할 3개와 기준 모델의 Presence 3개를 사용한다. 분할·Presence에 같은 면적 비율 모델을 쓰던 구성(S029)보다 holdout 오탐과 미탐이 줄어 역할을 분리했다. [역할 분리의 결과](../README.md#역할-분리-전후)를 참고한다. 이 6모델 앙상블 자체의 독립 OOF는 없으며, 구성 모델의 OOF로 이를 대신하지 않는다. [학습 명령과 재학습 가중치 조립](reproduction.md)을 제공한다.

## 다른 모델과 후반 실험

아래에는 추가 모델·학습 방법의 비교 결과를 요약했다. 이 비교에 사용한 외부 구현·가중치와 전체 실행 코드는 포함하지 않았다.

| 비교 | 실험 결과와 한계 |
|---|---|
| SegFormer-B0, 무작위 초기화 | 높은 학습률에서는 Presence 학습이 정체됐고, 낮춘 조건은 학습 손실이 감소해도 검증 손실이 증가했다. 선택 checkpoint의 최종 출력은 U-Net을 넘지 못했다. 사전학습 SegFormer의 성능 평가가 아니다. |
| HRNet small-v1 | 시험한 Adam·SGD 조건 모두 기준보다 Presence·Shape 진단이 낮았다. optimizer 비교에서는 예산·일정도 함께 달라 단일 원인으로 해석하지 않았다. |
| 외부 공개 해법 24조건 | 선택 checkpoint 중 기준 이상의 Presence와 더 높은 Shape를 동시에 만족한 조건이 없었다. 현재 입력으로 옮기는 과정에서 구조·전처리도 달라 원 해법의 일반적인 우열을 뜻하지 않는다. |
| 격자점·연속 좌표 학습 | 연속 좌표 방식은 격자점 대안보다 좋았지만 coverage 기준을 대체할 근거가 부족했다. 작은 양성 누락 감소와 오탐 증가가 함께 나타났다. |
| 학습 지역 확장·전체 재적합 | 로컬 개선과 플랫폼 결과가 일치하지 않았다. 정규화·학습량·선택 방식도 함께 달라 지역 수만의 효과로 해석하지 않는다. [후속 비교](results.md#후속-모델의-집계-비교) 참고. |

이 비교에 쓰인 개발 지역은 반복 사용됐다. 원자료를 사용한 전체 재학습과 새로운 지역 일반화는 미검증이며 플랫폼 점수도 다시 계산하지 않았다. [확인 범위](reproduction.md#실행-환경)를 참고한다.
