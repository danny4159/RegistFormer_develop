# IXI 데이터셋 전처리 파이프라인 (NIfTI → H5)

T1 = fixed, T2 = moving. IXI는 T1/T2 모두 1×1×1mm isotropic에 가까운 이상적인 스캔이라, rigid registration만으로도 거의 완벽하게 정렬되어 T2를 GT로 활용 가능. 이 정렬된 T2를 기준으로 misalignment / slice thickness / artifact를 시뮬레이션해서 "정렬된 T1 ↔ 흔들리거나 열화된 T2" 쌍을 만든다.

```
원본 NIfTI (T1, T2)
   │  SimpleITK rigid registration + resample + 정규화
   ▼
T1, T2                       (256,256,80)   ← 정렬된 원본
   │  Misalignment 시뮬레이션 (T2 → T2_moved)
   ▼
T2_moved                     (256,256,80)
   │  Slice thickness 시뮬레이션 (3mm / 5mm)
   ▼
T2_3mm, T2_5mm, T2_moved_3mm, T2_moved_5mm
   │  (선택) Artifact 주입
   ▼
T2_moved_{gibbs|kspike|motion_v2d|ghost}_{weak|medium|strong}
```

---

## 1. 원본 전처리

1. SimpleITK로 rigid registration (T1 기준 T2 정렬)
2. 1×1×1 mm isotropic resample, 공통 grid로 shape (256,256,100) 통일
3. intensity clip: [0, 99.9th percentile]
4. [-1, 1] 정규화

## 2. Z-trim + Background removal

- z축 양쪽 10 slice 제거 → (256,256,100) → (256,256,80)
- Otsu thresholding으로 foreground mask 생성 → slice별 hole filling → Gaussian feathering(σ=1.5) → `out = vol × mask_soft + (-1.0) × (1-mask_soft)`

결과: `T1, T2` — (256,256,80), 정렬된 원본.

## 3. Misalignment 시뮬레이션 (T2 → T2_moved)

**Rigid (torchio `RandomAffine`)**
- rotation ±5°, translation ±5 voxel (x/y/z 독립)
- 빈 공간은 Otsu 임계값으로 채움 (`default_pad_value='otsu'`)

**Non-linear (MONAI `Rand3DElastic`)**
- 보간: cubic spline (order=3)
- 패딩: reflect
- sigma=13, magnitude=650 (고정)
- random offset field(3,H,W,Z) → Gaussian smoothing(σ=13) → ×magnitude → sampling grid → cubic 재샘플링

이후 misalign으로 생긴 새 배경 영역에 background removal 재적용.

결과: `T2_moved` — (256,256,80).

## 4. Slice Thickness 시뮬레이션

목적: 임상에서 SNR 확보를 위해 두꺼운 slice로 촬영하는 경우를 재현.

대상: `T2`, `T2_moved` → 3mm/5mm 버전 생성.

1. z축 다운샘플 (`zoom_factor_z = 1/N`)
2. 원래 shape으로 업샘플
3. 둘 다 `scipy.ndimage.zoom`, order=3 (cubic)

결과: `T2_3mm, T2_5mm, T2_moved_3mm, T2_moved_5mm` — (256,256,80).

## 5. Artifact 주입 (T2_moved 대상)

4종류 × 3단계(weak/medium/strong):

| Artifact | 방법 |
|---|---|
| Gibbs ringing | MONAI `RandGibbsNoise` |
| K-space spike | MONAI `RandKSpaceSpikeNoise` |
| Motion (2D, per-slice) | FFT 기반 slice별 phase ramp + rotation, 궤적 3종(sudden/periodic/random-walk) |
| Ghosting | torchio `RandomGhosting` |

결과 그룹: `T2_moved_gibbs_{level}`, `T2_moved_kspike_{level}`, `T2_moved_motion_v2d_{level}`, `T2_moved_ghost_{level}`

---

## H5 파일 구조

- 파일: train/val/test 별 80-slice H5, 환자 수 train=30, val=3, test=30
- 그룹(= H5 최상위 key) 내부에 환자 ID(예: `IXI033-HH-1259`)로 dataset 저장

```
T1, T2
T2_moved
T2_3mm, T2_5mm
T2_moved_3mm, T2_moved_5mm
T2_moved_gibbs_{weak,medium,strong}
T2_moved_kspike_{weak,medium,strong}
T2_moved_motion_v2d_{weak,medium,strong}
T2_moved_ghost_{weak,medium,strong}
```

각 dataset: `shape=(256,256,80)`, `dtype=float32`, `range=[-1,1]`, gzip 압축.

---

## 구현 코드 경로

- `scripts/IXI_Ver4_Preprocess_Pipeline.py` — 원본 정렬 → misalignment → slice thickness 시뮬레이션 통합 파이프라인 (80-slice H5 생성)
- `scripts/generate_artifact_h5.py` — Artifact 주입 (Gibbs/K-space spike/Motion/Ghost)
- 원본 개발 당시 노트북(단계별): `Daniel_ssd2/Jupyter_notebook/IXI_preprocess/` (`[Preprocessing] IXI resampling + h5저장 코드.ipynb`, `(Preprocess_3) MisalignSimulation (IXI h5 misalign and 20SliceOut).ipynb` 등)
