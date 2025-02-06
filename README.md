# CMoE

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

Download the models from [Huggingface](https://huggingface.co/), then the you can run the code run_cmoe.py. Set model path as 'MODEL_PATH'.

You can run the pre-defined testing script 'run.sh' by:
```bash
bash run.sh
```

Or resetting the hyperparameters to run customized setting.
For example, run S2A2E16 with 2,048 fine-tuning data on wikitext2:
```python
python run_cmoe.py $MODEL_PATH wikitext2 \ 
--nshared 2 \
--nactivated 2 \
--nexperts 16 \
--nsamples 2048 \
--extra-lr 0.001 --bias-speed 0.001 --new-eval
```

## Evaluation

Our code automatically run ppl eval.
If you want to do evaluation on downstream tasks, you can add the arg `--eval-zero`, where the code is implemented by [Wanda](https://github.com/locuslab/wanda).

## Cite

If you found this work useful, please consider citing:

```
```
