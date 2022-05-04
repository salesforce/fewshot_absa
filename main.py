"""
Fine-tuning pretrained language model (GPT2) for Global Intent Model
"""

import argparse
import glob
import logging
import os
import pickle
import random
import re
import json

import numpy as np
import torch
from tqdm import tqdm, trange
import ipdb

from transformers import (
    WEIGHTS_NAME,
    # AdamW,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    # get_linear_schedule_with_warmup,
)

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer, GPT2Config
from model.modeling_gpt2 import GPT2LMHeadModel
from model.modeling_gpt2 import GPT2DoubleHeadsModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.modeling_utils import SequenceSummary

# comment this if you want to load gpt2 class from transformers
# from models import GPT2LMHeadModel
# from models import GPT2Config, GPT2SmallConfig

# uncomment this if you want to load gpt2 class from transformers
# from transformers import GP2Config, GPT2LMHeadModel

from data.dataset.language_model import *
from util.model import *
from util.model import _sorted_checkpoints
from util.language_model import get_optimizer_scheduler
from util.gpt2_args_parser import ArgsParser
from util.metrics import *
from torch.nn import CrossEntropyLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

logging.getLogger('transformers.generation_utils').disabled = True



MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2_double": (GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True


def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

        # modify head head with for 2 classes
        if args.model_type == 'gpt2_double':
            if 'checkpoint' not in args.model_name_or_path:
                config.num_labels = 2
                config.summary_type = 'last'
                model.multiple_choice_head = SequenceSummary(config)
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)



    model.to(args.device)

    if args.model_name_or_path == 'openai-gpt':
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    elif args.model_name_or_path == 'gpt2':
        pass

    return model, tokenizer, model_class, args


def get_training_info(dataloader, args):
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
    return global_step, epochs_trained, steps_trained_in_current_epoch


def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, tr_loss_lm, logging_loss_lm, tr_loss_mc, logging_loss_mc, global_step,
                steps_trained_in_current_epoch, tb_writer, args, best_checkpoint_acc, tr_loss_label, logging_loss_label, logged_weights, initial_weights):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        if 'gpt2' in args.model_type:
            if args.model_type == 'gpt2':
                inputs, labels = (batch, batch)
            elif args.model_type == 'gpt2_double':
                inputs, labels = batch[0], batch[0]
                mc_labels = batch[1]
                mc_labels = mc_labels.to(args.device)
        else:
            inputs, labels = batch[0], batch[1]

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        # ipdb.set_trace()
        model.train()
        if args.model_type == 'gpt2_double':
            outputs = model(inputs, labels=labels, mc_labels=mc_labels)
            loss_lm = outputs[0]
            loss_mc = outputs[1]
            loss = loss_lm + loss_mc
        else:
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            step_loss_label = compute_label_position_loss(logits, inputs, labels, tokenizer, args)


        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.model_type == 'gpt2_double':
                loss_lm = loss_lm.mean()
                loss_mc = loss_mc.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            step_loss_label = step_loss_label / args.gradient_accumulation_steps
            if args.model_type == 'gpt2_double':
                loss_lm = loss_lm / args.gradient_accumulation_steps
                loss_mc = loss_mc / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        tr_loss_label += step_loss_label.item()

        if args.model_type == 'gpt2_double':
            tr_loss_lm += loss_lm.item()
            tr_loss_mc += loss_mc.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            results = {}
            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if (
                        args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well

                    for split in args.eval_splits:
                        results = evaluate(args, model, tokenizer, split=split)
                        for key, value in results.items():
                            tb_writer.add_scalar("metrics/{}/{}".format(split, key), value, global_step)

                    model_result = {}
                    model_result = compute_gpt2_weights_norm(model, model_result)

                    for key, value in model_result.items():
                        tb_writer.add_scalar("model/{}".format(key), value, global_step)

                    weights = extract_gpt2_weights(model)
                    for k in weights.keys():
                        aggregate_shift = np.mean(np.abs(weights[k] - logged_weights[k]))
                        aggregate_shift_percentage = np.mean(
                            np.abs(weights[k] - logged_weights[k]) / np.abs(initial_weights[k]))
                        tb_writer.add_scalar(f"model/aggregate_shift_{k}", aggregate_shift, global_step)
                        tb_writer.add_scalar(f"model/aggregate_shift_percentage_{k}", aggregate_shift_percentage, global_step)
                        logged_weights[k] = weights[k]

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                if args.model_type == 'gpt2_double':
                    tb_writer.add_scalar("loss_lm", (tr_loss_lm - logging_loss_lm) / args.logging_steps, global_step)
                    tb_writer.add_scalar("loss_mc", (tr_loss_mc - logging_loss_mc) / args.logging_steps, global_step)
                # else:
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                if (tr_loss_label - logging_loss_label) == 0:
                    ipdb.set_trace()
                tb_writer.add_scalar("loss_label", (tr_loss_label - logging_loss_label) / args.logging_steps, global_step)
                logging_loss = tr_loss
                logging_loss_lm = tr_loss_lm
                logging_loss_mc = tr_loss_mc
                logging_loss_label = tr_loss_label

            # # save checkpoint
            step_metric = 'corr' if args.subtask == 'cola' else 'acc'
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.evaluate_during_training:
                    step_performance = None
                    for k, v in results.items():
                        if step_metric in k:
                            step_performance = v
                    if step_performance is None:
                        step_performance = results['loss']
                        step_metric = 'loss'

                    if step_metric is not 'loss' and step_performance > best_checkpoint_acc:
                        logger.info(f"found a better checkpoint @ {global_step}: acc={step_performance}")
                        best_checkpoint_acc = step_performance
                        save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step)
                    else:
                        logger.info(f"found a better checkpoint @ {global_step}: loss={step_performance}")
                        best_checkpoint_acc = step_performance
                        save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss, tr_loss_lm, logging_loss_lm, tr_loss_mc, logging_loss_mc, best_checkpoint_acc, tr_loss_label, logging_loss_label, logged_weights


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/{}'.format('/'.join(args.output_dir.split('/')[1:])))

    # Prepare dataloader
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args)

    # total iteration and batch size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    # compute model params norm before training
    model_result = {}
    model_result = compute_gpt2_weights_norm(model, model_result)
    for key, value in model_result.items():
        tb_writer.add_scalar("model/{}".format(key), value, 0)

    logged_weights = extract_gpt2_weights(model)
    initial_weights = extract_gpt2_weights(model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(train_dataset)))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))

    global_step, epochs_trained, steps_trained_in_current_epoch = get_training_info(train_dataloader, args)

    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_lm, logging_loss_lm = 0.0, 0.0
    tr_loss_mc, logging_loss_mc = 0.0, 0.0
    tr_loss_label, logging_loss_label = 0.0, 0.0
    best_checkpoint_acc = 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for _ in train_iterator:

        model, optimizer, scheduler, global_step, tr_loss, logging_loss, tr_loss_lm, logging_loss_lm, tr_loss_mc, logging_loss_mc, best_checkpoint_acc, tr_loss_label, logging_loss_label, logged_weights = train_epoch(
            model, tokenizer, optimizer,
            scheduler, train_dataloader,
            tr_loss, logging_loss,
            tr_loss_lm, logging_loss_lm,
            tr_loss_mc, logging_loss_mc,
            global_step,
            steps_trained_in_current_epoch,
            tb_writer, args, best_checkpoint_acc, tr_loss_label, logging_loss_label, logged_weights, initial_weights)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", split='val'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, split=split)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare dataloader
    eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split=split)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} {} *****".format(split, prefix))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Batch size = {}".format(args.eval_batch_size))
    logger.info(" Confidence threshold = {}".format(args.confidence_threshold))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    total_pred = []
    total_ground = []
    result = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        if 'gpt2' in args.model_type:
            if args.model_type == 'gpt2':
                inputs, labels = (batch, batch)
            elif args.model_type == 'gpt2_double':
                inputs, labels = batch[0], batch[0]
                mc_labels = batch[1]
                mc_labels = mc_labels.to(args.device)
        elif 't5' in args.model_type:
            inputs, labels = batch[0], batch[1]
        else:
            raise TypeError('unknown model type')
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            lm_loss = 0
            if 'gpt2' in args.model_type:
                if args.subtask == 'single_term_polarity':
                    batch_pred, batch_ground = compute_single_term_polarity(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'aspect_term':
                    batch_pred, batch_ground = compute_aspect_term(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'single_category_polarity':
                    batch_pred, batch_ground = compute_single_category_polarity(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'aspect_category':
                    batch_pred, batch_ground = compute_aspect_category(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'aspect_term_aspect_category':
                    batch_pred, batch_ground = compute_aspect_term_aspect_category(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'sentiment':
                    batch_pred, batch_ground = compute_sentiment(model, inputs, labels, tokenizer, args)
                elif args.subtask == 'sst2':
                    batch_pred, batch_ground = compute_sst2(model, inputs, labels, tokenizer, args)
            else:
                raise TypeError("unknown subtask")

            total_pred.extend(batch_pred)
            total_ground.extend(batch_ground)
            eval_loss += lm_loss
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result['loss'] = eval_loss
    result['perplexity'] = perplexity

    if 'gpt2' in args.model_type:
        if args.subtask == 'single_term_polarity':
            result = compute_term_polarity_metrics(result, labels=total_ground,
                                     predictions=total_pred)
        elif args.subtask == 'aspect_term':
            result = compute_aspect_term_metrics(result, labels=total_ground,
                                                   predictions=total_pred)
        elif args.subtask == 'single_category_polarity':
            result = compute_category_polarity_metrics(result, labels=total_ground,
                                     predictions=total_pred)
        elif args.subtask == 'aspect_category':
            result = compute_aspect_category_metrics(result, labels=total_ground,
                                                   predictions=total_pred)
        elif args.subtask == 'aspect_term_aspect_category':
            result = compute_aspect_term_aspect_category_metrics(result, labels=total_ground,
                                                   predictions=total_pred)
        elif args.subtask == 'sentiment':
            result = compute_sentiment_metrics(result, labels=total_ground,
                                                   predictions=total_pred)
        elif args.subtask == 'sst2':
            result = compute_sst2_metrics(result, labels=total_ground,
                                                   predictions=total_pred)
    else:
        raise TypeError('unknown subtask')

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_{}.txt".format(split))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if 'term' in args.subtask:
        subtask_st = '<|term|>'
        subtask_et = '<|endofterm|>'
    elif 'category' in args.subtask:
        subtask_st = '<|category|>'
        subtask_et = '<|endofcategory|>'
    elif 'sentiment' in args.subtask:
        subtask_st = '<|sentiment|>'
        subtask_et = '<|endofsentiment|>'
    elif 'sst2' in args.subtask:
        subtask_st = '<|sentiment|>'
        subtask_et = '<|endofsentiment|>'
    else:
        raise TypeError('unknown subtask')

    # write generated output
    output_generation_file = os.path.join(eval_output_dir, prefix, "eval_generations_{}.txt".format(split))
    with open(output_generation_file, "w") as writer:
        if 'gpt2' in args.model_type:
            if args.subtask == 'aspect_term_aspect_category':
                for l_p, l_t in zip(total_pred, total_ground):
                    true_term = l_t.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()
                    pred_term = l_p.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()

                    true_category = l_t.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()
                    pred_category = l_p.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()

                    true_term_sorted, pred_term_sorted = sort_output(true_term, pred_term)
                    true_category_sorted, pred_category_sorted = sort_output(true_category, pred_category)

                    true_joint = f"<|term|> {true_term_sorted} <|endofterm|> <|category|> {true_category_sorted} <|endofcategory|>"
                    pred_joint = f"<|term|> {pred_term_sorted} <|endofterm|> <|category|> {pred_category_sorted} <|endofcategory|>"
                    correct = 'True' if pred_joint == true_joint else 'False'
                    writer.write(f"{l_p}\t|\t{pred_joint}\t|\t{true_joint}\t|\t{correct}\n")
            else:
                if 'aspect' not in args.subtask:
                    for l_p, l_t in zip(total_pred, total_ground):
                        true_target = l_t.split(subtask_st)[-1].split(subtask_et)[0].strip()
                        pred_target = l_p.split(subtask_st)[-1].split(subtask_et)[0].strip()
                        correct = 'True' if pred_target == true_target else 'False'
                        writer.write(f"{l_p}\t|\t{pred_target}\t|\t{true_target}\t|\t{correct}\n")
                else:
                    for l_p, l_t in zip(total_pred, total_ground):
                        true_term = l_t.split(subtask_st)[-1].split(subtask_et)[0].strip()
                        pred_term = l_p.split(subtask_st)[-1].split(subtask_et)[0].strip()
                        true_term_sorted, pred_term_sorted = sort_output(true_term, pred_term)
                        correct = 'True' if pred_term_sorted == true_term_sorted else 'False'
                        writer.write(f"{l_p}\t|\t{pred_term_sorted}\t|\t{true_term_sorted}\t|\t{correct}\n")
        else:
            raise TypeError('unknown model type')


    return result


def main():
    args = ArgsParser().parse()


    args.eval_splits = args.eval_splits.strip().split(',')
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            logger.info("--should_continue is true, but no checkpoint found in --output_dir")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if args.no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", index=args.device)
        args.n_gpu = 1
    else:  # initialize distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # set random seed
    set_seed(args)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, args = get_model_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    logger.info("Training/evaluation parameters {}".format(args))

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # only first process will preprocess data/caching

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()  # end of barrier

        global_step, train_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = {}, average loss = {}".format(global_step, train_loss))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.eval_checkpoint]

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            for split in args.eval_splits:
                result = evaluate(args, model, tokenizer, prefix=prefix, split=split)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

    return results


if __name__ == "__main__":
    main()
