# RegistFormer

## This code is being refactored and developed.

## 1. Description

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking


## 2. Installation

```bash
Will be uploaded soon
```

## 3. Dataset
#### Dataset 
Grand challenge ['SynthRAD 2023'](https://synthrad2023.grand-challenge.org/) Pelvis MR, CT

#### Preprocessing
- MR: 
  - N4 correction 
  - Nyul Histogram Matching 
  - z-score norm each patient 
  - -1~1 minmax norm each patient

- CT: 
  - 5%, 95% percentile clip 
  - z-score norm whole patient 
  - -1 ~ 1 minmax norm whole patient

#### File Format: 
h5

#### Dataset download
https://drive.google.com/drive/folders/19a9VF9TYMyg6TAnOyRokn4d46_Nfhvfa?usp=sharing

Download this preprocessed MR to CT dataset, and insert it into the 'data/SynthRAD_MR_CT_Pelvis' folder.


#### Pretrained model download
https://drive.google.com/drive/folders/1dR1kGKsZQCLMtXnNqJ8Arm5aFl2IslrX?usp=sharing

Download this pretrained model, and insert it into the 'pretrained' folder.


## 4. How to run

```bash
#### Training
python registformer/train.py model='registformer.yaml' trainer.devices=[0] tags='SynthRAD_Registformer_try'
```