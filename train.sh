

export GPU=$1
export MODEL=$2
export MODEL_NAME=$3
export TRAIN_DATASET=$4
export VAL_DATASET=$5
export SUBTASK=$6
export BATCH=$7
export BLOCK=${8:-128}
export LOG_FREQ=${9:-1000}
export EPOCHS=${10:-200000}
export SEED=${11:-42}
export LR=${12:-5e-5}
export WARM=${13:-0}

export TRAIN_FILE=./resources/gpt2/${TRAIN_DATASET}.txt
export VAL_FILE=./resources/gpt2/${VAL_DATASET}.txt

export OUTPUT=output/generative/${SUBTASK}/${TRAIN_DATASET}/${MODEL}/${MODEL_NAME}_block${BLOCK}_batch${BATCH}_lr${LR}_warm${WARM}_epochs${EPOCHS}_log${LOG_FREQ}_seed${SEED}


CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --dataset=$TRAIN_DATASET \
    --subtask=$SUBTASK \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$VAL_FILE \
    --test_data_file=$TEST_FILE \
    --evaluate_during_training \
    --save_steps $LOG_FREQ \
    --logging_steps $LOG_FREQ \
    --per_gpu_train_batch_size $BATCH \
    --per_gpu_eval_batch_size $BATCH \
    --num_train_epochs $EPOCHS \
    --should_continue \
    --block_size $BLOCK \
    --learning_rate $LR \
    --warmup_steps $WARM \
    --seed $SEED \
    --eval_splits val
