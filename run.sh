export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=""

python run_cmoe.py $MODEL_PATH wikitext2 --new-eval --nshared 2 --nactivated 2 --nexperts 16 --nsamples 64 --extra-lr 0.001 --bias-speed 0.001
