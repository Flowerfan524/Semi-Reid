GN=0
if [ "$1" != "" ];then
    GN=$1
fi
source activate py36
echo "Test on GPU: $GN"
CUDA_VISIBLE_DEVICES=$GN python spaco.py
