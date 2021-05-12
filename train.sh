if [ -z "$1" ]; then
    echo "Usage: ./run.sh gpu_id"
else
    export CUDA_VISIBLE_DEVICES=$1
    python -m src.train
fi
