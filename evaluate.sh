DATASET=market1501std
CHECKPOINT=$1
COMBINE=123

CUDA_VISIBLE_DEVICES=1 python evaluate.py -d $DATASET -c $CHECKPOINT --combine $COMBINE --single-eval
