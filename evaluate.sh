


export GPU=$1
export TEST_DATASET=$2
export SUBTASK=$3
export CHECKPOINT=$4

export TEST_FILE=./resources/gpt2/${TEST_DATASET}.txt


CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --subtask=$SUBTASK \
    --checkpoint_dir=$CHECKPOINT \
    --do_eval \
    --test_data_file=$TEST_FILE \
    --eval_splits test
