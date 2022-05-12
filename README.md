## A generative languege model for fewshot aspect based sentiment analysis
This repository contains the code of our [paper](https://arxiv.org/abs/2204.05356) which is published in 
*Findings of NAACL 2022*. 




#### 1. Install and preprocess data for language modeling 
to install required packages and preprocesss the data, run below commands. It preprocess data and create training data for full-shot and fewshot training of autoregressive language model.

    pip install -r requirements.txt
    cd preprocess && bash preprocess_data.sh

#### 2. train language model
below is a sample training script for training on aspect term polarity on semeval14 dataset.

    export GPU=0
    export MODEL=gpt2
    export MODEL_NAME=gpt2
    export TRAIN_DATASET=semeval14_restaurants_aspect_term_train
    export VAL_DATASET=semeval14_restaurants_aspect_term_val
    export SUBTASK=single_term_polarity
    export BATCH=8
    export BLOCK=128
    
    bash train.sh $GPU 
                  $MODEL 
                  $MODEL_NAME 
                  $TRAIN_FILENAME 
                  $TASK 
                  $BATCH 
                  $BLOCK 

#### 3. evaluate
below is the code for running evaluation for some checkpoints
    
    
    export GPU=0
    export TEST_FILE=semeval14_restaurant_aspect_term_test
    export SUBTASK=single_term_polarity
    export CHECKPOINT=/path_to_checkpoints
    
    bash evaluate.sh $GPU
                     $TEST_FILE
                     $SUBTASK
                     $CHECKPOINT
                      
         


 #### 4- demo
run a pretrained language model to generate aspect terms and aspect categories with their corresponding polarities 

    ipython notebook demo.ipynb
