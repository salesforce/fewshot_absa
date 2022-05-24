# A generative languege model for few-shot aspect-based sentiment analysis
Authors: [Ehsan Hosseini-Asl](https://scholar.google.com/citations?user=I9w3ON4AAAAJ&hl=en), [Wenhao Liu](https://scholar.google.com/citations?user=BaRpQ_kAAAAJ&hl=en), [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en) 


This repository contains the code of our [paper](https://arxiv.org/abs/2204.05356) which is published in 
**Findings of NAACL 2022**. 

Our proposed approach is based on autoregressive language modeling. Given the input review sentence, the model generates all aspect terms, aspect categories and their corresponding polarities in a single sequence generation.

![Example 1:,](/images/exp1.gif)

![Example 2:](/images/exp2.gif)

![Example 3:](/images/exp3.gif)


## Introduction
Sentiment analysis is an important task in natural language processing. In recent works, pre-trained language models are often used to achieve state-of-the-art results, especially when training data is scarce. It is common to fine-tune on the downstream task, usually by adding task-specific layers on top of the model. 

In this paper, we focus on aspect-based sentiment analysis, which involves extracting aspect term, category, and predicting their corresponding polarities. In particular, we are interested in few-shot settings. We propose to reformulate the extraction and prediction tasks into the sequence generation task, using a generative language model with unidirectional attention (GPT2 is used unless stated otherwise). This way, the model learns to accomplish the tasks via language generation without the need of training task-specific layers. 

Paper link: https://arxiv.org/abs/2204.05356


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
    - [Preprocessinng](#preprocessing)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Demo](#demo)
- [Citation](#citation)
- [License](#license)

### Installation

The package general requirements are

- Python >= 3.6
- Pytorch >= 1.7.1 (installation instructions [here](https://pytorch.org/))
- Transformers >= 4.0.0 (installation instructions [here](https://huggingface.co/transformers/))
 
1- The package can be installed by running the following command.  

    pip install -r requirements.txt

2- Running inside docker container

    docker build -t <image_name>:<tag> -f Dockerfile
    
## Usage
This section explains steps to preprocess [Semeval14 Task 4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) and [Semeval16 Task 5](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) dataset and training the model. 

### Preprocessing 
First, create a directory named `resources` and download [Semeval14 Task 4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) and [Semeval16 Task 5](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) and move them to the ```resources``` directory. There should be two subdirectories `semeval14` and `semeval16`. 

Then, run below commands to preprocess raw data and create training data for full-shot and fewshot training of autoregressive language model.


    cd preprocess && bash preprocess_data.sh

Each training example will be represented as a sequence, which contains review sentence with corresponding aspect term, aspect category and their polarities
```
<|endoftext|> <|review|> they did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), 
below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. <|endofreview|> 
<|term|> toast negative , mayonnaise negative , bacon negative , cheese neutral , ingredients negative , plate neutral , omelet neutral <|endofterm|> 
<|category|> food negative <|endofcategory|> <|endoftext|>
```


### Training
below is a sample training script for training on aspect term polarity on semeval14 dataset.

    export GPU=0
    export MODEL=gpt2
    export MODEL_NAME=gpt2
    export TRAIN_DATASET=semeval14_restaurants_aspect_term_train
    export VAL_DATASET=semeval14_restaurants_aspect_term_val
    export SUBTASK=single_term_polarity
    export BATCH=8
    export BLOCK=128
    
    bash train.sh $GPU \
                  $MODEL \ 
                  $MODEL_NAME \
                  $TRAIN_FILENAME \
                  $VAL_FILENAME \
                  $SUBTASK \
                  $BATCH \
                  $BLOCK 


 
### Evaluation
below is the code for running evaluation for some checkpoints
    
    
    export GPU=0
    export TEST_FILE=semeval14_restaurant_aspect_term_test
    export SUBTASK=single_term_polarity
    export CHECKPOINT=/path_to_checkpoints
    
    bash evaluate.sh $GPU \
                     $TEST_FILE \
                     $SUBTASK \
                     $CHECKPOINT
                      
        
        
         
### Demo
run a pretrained language model to generate aspect terms and aspect categories with their corresponding polarities 


    ipython notebook demo.ipynb


### Citation

```
@article{hosseini2022generative,
  title={A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis},
  author={Hosseini-Asl, Ehsan and Liu, Wenhao and Xiong, Caiming},
  journal={arXiv preprint arXiv:2204.05356},
  year={2022}
}
```

### License

The code is released under the BSD-3 License - see [LICENSE](LICENSE.txt) for details

