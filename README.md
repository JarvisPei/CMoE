# CMoE: Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis

Official implementation of **CMoE**, an analytical post-training framework that rapidly restructures dense FFNs into sparse MoE architectures using only a small calibration dataset.

## News

- **[2026.04]** Updated arXiv preprint with the camera-ready title: [Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](https://arxiv.org/abs/2502.04416).
- **[2026.04]** Our paper **"Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis"** has been accepted to **ACL 2026 Main Conference** (recommended for oral presentation).
- **[2025.02]** Initial preprint released: [CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference](https://arxiv.org/abs/2502.04416v1).

## Overview

CMoE analyzes neuron activation patterns to partition FFN neurons into:

- **Shared experts** — high-frequency neurons that are always active.
- **Routed experts** — low-frequency neurons grouped by co-activation, activated conditionally per token.

A router is then constructed **analytically** from representative neuron statistics, avoiding expensive router training and enabling immediate deployment with optional lightweight fine-tuning (2k samples).


## Dependencies

```bash
conda create -n cmoe python=3.11
conda activate cmoe
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install datasets==2.21.0
pip install transformers==4.47.1
pip install accelerate==1.2.1
pip install sentencepiece==0.2.0
pip install protobuf==5.29.2
pip install matplotlib==3.10.0
pip install lap==0.5.12
pip install peft==0.14.0
```

Note: please modify the version of some packages for your own environment.

## Quick Start

Download the models from [Hugging Face](https://huggingface.co/), then set the model path as `MODEL_PATH` and run the pre-defined script:

```bash
bash run.sh
```

Or configure hyperparameters for a custom setting. For example, to run **S2A2E16** (2 shared + 2 active routed / 16 total experts) with 2,048 WikiText-2 fine-tuning samples:

```bash
python run_cmoe.py $MODEL_PATH wikitext2 \
  --nshared 2 \
  --nactivated 2 \
  --nexperts 16 \
  --nsamples 2048 \
  --extra-lr 0.001 --bias-speed 0.001 --new-eval
```

Key arguments:

- `--nshared`: number of shared experts.
- `--nactivated`: number of routed experts activated per token.
- `--nexperts`: total number of experts.
- `--nsamples`: number of fine-tuning samples (set to 0 for training-free mode).

## Evaluation

The code automatically runs perplexity evaluation on the calibration dataset.
For zero-shot downstream tasks (PIQA, WinoGrande, ARC-E/C, HellaSwag, MMLU, etc.), add `--eval-zero`. The zero-shot evaluation implementation is adapted from [Wanda](https://github.com/locuslab/wanda).

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{pei2026analytical,
  title={Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis},
  author={Pei, Zehua and Zhen, Hui-Ling and Zou, Lancheng and Yu, Xianzhi and Liu, Wulong and Pan, Sinno Jialin and Yuan, Mingxuan and Yu, Bei},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={4777--4789},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.