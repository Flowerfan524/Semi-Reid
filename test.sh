GPU=$1
RE=$2
MODEL=parallel_spaco
CHECKPOINT=logs/$MODEL/$RE


for ((seed=1; seed<=1; seed++))
do
        bash evaluate.sh $GPU "${CHECKPOINT}/seed_${seed}"
done
