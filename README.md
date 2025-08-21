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

- Place datasets under `datasets/` as described in the original Multimodal-Infomax repo.
- Example: `datasets/MOSEI/mosei_senti_data_noalign.pkl`

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

If you use this code, please cite:

```
@article{your2025btw,
  title={BTW: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration},
  author={Your Name and others},
  journal={...},
  year={2025}
}
```# MultiModal-InfoMax

This repository contains the official implementation code of the paper [Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2109.00412.pdf), accepted at **EMNLP 2021**.

:fire:  If you would be interested in other multimodal works in our DeCLaRe Lab, welcome to visit the [clustered repository](https://github.com/declare-lab/multimodal-deep-learning)

## Introduction
Multimodal-informax (MMIM) synthesizes fusion results from multi-modality input through a two-level mutual information (MI) maximization. We use BA (Barber-Agakov) lower bound and contrastive predictive coding as the target function to be maximized. To facilitate the computation, we design an entropy estimation module with associated history data memory to facilitate the computation of BA lower bound and the training process.

![Alt text](img/ModelFigSingle.png?raw=true "Model")

## Usage
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