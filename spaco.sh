DATASET=market1501std
ARCH1=resnet50
ARCH2=densenet121
BATCH_SIZE=32
LOGSDIR=logs

python examples/spaco.py \
        --dataset $DATASET \
        --arch1 $ARCH1 \
        --arch2 $ARCH2 \
        --batch-size $BATCH_SIZE \
        --logs-dir $LOGSDIR
