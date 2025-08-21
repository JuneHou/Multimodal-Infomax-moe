# BTW: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration

This repository contains the official code for our paper:

**BTW: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration**

## Paper

> [BTW: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration](#)  
> *Your Name*, et al.  
> [arXiv/Journal link here]

## Overview

This codebase is built upon [Multimodal-Infomax](https://github.com/declare-lab/Multimodal-Infomax) and provides:
- The original baseline implementation.
- Our proposed method (BTW) with variance stabilization.

## Environment Setup

We recommend using conda:

```bash
conda create -n btw_env python=3.8
conda activate btw_env
pip install -r requirements.txt
```

## Datasets
The dataset downloading and preprocessing follows the [Multimodal-Infomax](https://github.com/declare-lab/Multimodal-Infomax)

1. Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1Wxo4Bim9JhNmg8265p3ttQ) (extraction code: g3m2). Place them under the folder `Multimodal-Infomax/datasets`

2. Set up the environment (need conda prerequisite)
```
conda env create -f environment.yml
conda activate MMIM
```

3. Start training
```
python main.py --dataset mosi --contrast
```

## Running Experiments

### Baseline

To run the original baseline (using `model.py` and `solver.py`):

```bash
bash run_baseline.sh
```

### BTW (Ours)

To run our method (using `model_btw.py` and `solver_btw.py`):

```bash
bash run_btw.sh
```

## Example Command

```bash
python src/main.py --dataset mosi --modality text_audio_video --num_modality 3 --n_class 1 --moe_experts 16 --lr_main 1e-4 --out_folder "path/to/your/folder" --weights_type kl --kl_type residual --seed 1111
```


## Replication on MIMIC-IV with Fuse-MoE

For experiments and replication of Fuse-MoE with the multimodal MIMIC-IV dataset, please see our companion repository:

[https://github.com/JuneHou/Multimodal-Transformer.git](https://github.com/JuneHou/Multimodal-Transformer.git)

## Citation
Please cite our paper if you find our work useful for your research:

```
@inproceedings{
hou2025btw,
title={{BTW}: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration},
author={Jun Hou and Le Wang and Xuan Wang},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=EXp1qDqhCk}
}
```