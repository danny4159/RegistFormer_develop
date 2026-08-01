# MRBrainS18 데이터셋 전처리 파이프라인 (NIfTI → H5)

FLAIR = fixed, T1(/IR) = moving. MRBrainS18은 IXI와 달리 모달리티 간 스캔이 실제로 서로 다른 시점/해상도로 촬영되어 원본부터 자연스러운 misalignment가 존재 → 인위적 misalignment 시뮬레이션 없이, **실제 misalignment를 SimpleITK rigid registration으로 학습 시점에 정합**하는 방식.

```
원본 NIfTI (FLAIR, T1, IR, segm)  — 환자별 orig/ 폴더
   │  T1 → FLAIR 그리드로 resample (nibabel resample_from_to)
   ▼
FLAIR, resample_T1, IR, segm      (동일 grid)
   │  모달리티별 [-1,1] min-max 정규화 → 256×256 zero-pad
   ▼
FLAIR, T1, IR, Seg                (256,256,48)  → H5 저장 (오프라인, 1회)
   │  (학습 시점) SimpleITK rigid registration: T1(/IR) → FLAIR
   ▼
FLAIR(fixed) ↔ registered T1(moving)  — 매 학습 실행마다 캐싱되어 사용
```

---

## 1. H5 생성 (오프라인, Jupyter notebook)

**Resample**: 원본 T1은 FLAIR/IR과 grid가 다르므로 `nibabel.processing.resample_from_to(T1, FLAIR)`로 FLAIR 그리드에 맞춰 `resample_T1.nii.gz` 생성.

**정규화**: 모달리티별 독립 min-max → `2 * (x - min) / (max - min) - 1`

**패딩**: 원본 (240,240,48) → 상하좌우 8px씩 zero-pad → (256,256,48)
- FLAIR/T1/Seg: -1로 패딩
- IR: 배경이 -1이 아니라서 좌표 (10,10) 픽셀 값으로 패딩 (경계 불연속 방지)

**Split** (patient ID 기준, 총 30명):
- train: 24명 (`01,02,04,...,30`)
- val: 1명 (`15`)
- test: 6명 (`03,09,15,23,26,28`) — 15는 val/test 양쪽에 포함

결과: `train/trainset.h5`, `val/valset.h5`, `test/testset.h5`, 각 그룹 `FLAIR / T1 / IR / Seg`.

## 2. Registration (온라인, 학습 시점)

H5에는 misalign 시뮬레이션이 없고 원본 그대로 저장되어 있음. 대신 config에서:

```yaml
apply_linear_registration: True       # SimpleITK rigid registration
apply_non_linear_registration: False  # (선택) ConvexAdam/Anatomix non-linear, rigid 결과를 moving으로 사용
registration_targets: [2, 3]          # data_group_1(FLAIR) 기준, group_2/3(T1)을 등록
use_misalign_simul: False             # 인위적 misalign 없음 — 원본 자체가 misaligned
```

`src/data/components/transforms.py::_register_3d_rigid`:
- fixed=FLAIR, moving=T1(/IR)
- z축 20 slice reflect-edge 패딩 (경계 대각선 아티팩트 방지) 후 registration, 이후 crop
- SimpleITK `Euler3DTransform`, Mattes Mutual Information, RegularStepGradientDescent, multi-resolution(shrink 4/2/1, smoothing 2/1/0)
- linear interpolation, 배경 -1로 resample
- 환자별로 1회 계산되어 캐싱(`reg_cache`) 후 학습 내내 재사용

---

## H5 파일 구조

- 경로: `data/MRBrainS18/{train,val,test}/{split}set.h5`
- 그룹: `FLAIR, T1, IR, Seg` (Seg = segmentation mask)
- 환자 ID를 key로 저장 (예: `01`, `15`)
- 각 dataset: `shape=(256,256,48)`, `dtype=float32`, `range=[-1,1]` (Seg는 라벨 값)

---

## 구현 코드 경로

- H5 생성(오프라인, resample + 정규화 + 패딩): `Daniel_ssd2/Jupyter_notebook/MRBrainS18_preprocess/(Preprocessing) MRBrainS18 dataset nii -> h5 .ipynb`
- T1 → FLAIR resample: `Daniel_ssd2/Jupyter_notebook/MRBrainS18_preprocess/(Resample) save resample T1 in MRBrainS18 dataset.ipynb`
- Rigid registration(온라인, 학습 시점 캐싱): `src/data/components/transforms.py` (`_register_3d_rigid`, `MedicalImageDataset.__init__`의 registration cache 로직)
- Config: `configs/data/MRBrainS18_FLAIR_T1_IR.yaml`
