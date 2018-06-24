GPU=$1
REGULARIZER=$2
GAMMA=0.3
ITER_STEPS=4
DATASET=market1501std

for ((i=1; i<=10; i++))
do 
        echo "run parallel spaco with $REGULARIZER term, seed $i";
        CUDA_VISIBLE_DEVICES=$GPU python p_spaco.py --dataset $DATASET --seed $i --gamma $GAMMA --iter_steps $ITER_STEPS --regularizer $REGULARIZER
done
