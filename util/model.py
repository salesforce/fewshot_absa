
import torch
import glob
import random
import numpy as np
from typing import Dict, List, Tuple
import os
import shutil
import regex as re
import logging
logger = logging.getLogger(__name__)
import ipdb


def compute_label_position_loss(logits, inputs, labels, tokenizer, args):
    loss_label = None
    negative_token = tokenizer.encode(' negative')
    positive_token = tokenizer.encode(' positive')
    neutral_token = tokenizer.encode(' neutral')
    conflict_token = tokenizer.encode(' conflict')
    inparr = inputs.cpu().numpy()
    negative_row, negative_column = np.where(inparr == negative_token)
    negative_column = negative_column - 1  # shift position
    negative_logits = logits[negative_row, negative_column]
    positive_row, positive_column = np.where(inparr == positive_token)
    positive_column = positive_column - 1  # shift position
    positive_logits = logits[positive_row, positive_column]
    neutral_row, neutral_column = np.where(inparr == neutral_token)
    neutral_column = neutral_column - 1  # shift position
    neutral_logits = logits[neutral_row, neutral_column]
    conflict_row, conflict_column = np.where(inparr == conflict_token)
    conflict_column = conflict_column - 1  # shift position
    conflict_logits = logits[conflict_row, conflict_column]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    positive_labels = labels[np.where(labels.cpu().numpy() == positive_token)]
    loss_positive = loss_fct(positive_logits, positive_labels)
    negative_labels = labels[np.where(labels.cpu().numpy() == negative_token)]
    loss_negative = loss_fct(negative_logits, negative_labels)
    neutral_labels = labels[np.where(labels.cpu().numpy() == neutral_token)]
    loss_neutral = loss_fct(neutral_logits, neutral_labels)
    conflict_labels = labels[np.where(labels.cpu().numpy() == conflict_token)]
    loss_conflict = loss_fct(conflict_logits, conflict_labels)
    num_losses = 0
    step_loss_label = torch.tensor(0, dtype=torch.float64).to(args.device)
    if not loss_positive.isnan():
        step_loss_label += loss_positive
        num_losses += 1
    if not loss_negative.isnan():
        step_loss_label += loss_negative
        num_losses += 1
    if not loss_neutral.isnan():
        step_loss_label += loss_neutral
        num_losses += 1
    if not loss_conflict.isnan():
        step_loss_label += loss_conflict
        num_losses += 1
    if num_losses != 0:
        step_loss_label /= num_losses

    return step_loss_label


def compute_loss_at_initialization(model, data_iterator, tokenizer, args, tb_writer):
    total_loss = 0
    total_loss_label = 0
    steps = 0
    with torch.no_grad():
        for step, batch in enumerate(data_iterator):
            steps += 1
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.eval()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            logits = outputs[1]
            negative_token = tokenizer.encode(' negative')
            positive_token = tokenizer.encode(' positive')
            neutral_token = tokenizer.encode(' neutral')
            inparr = inputs.cpu().numpy()
            negative_row, negative_column = np.where(inparr == negative_token)
            negative_column = negative_column - 1  # shift position
            negative_logits = logits[negative_row, negative_column]
            positive_row, positive_column = np.where(inparr == positive_token)
            positive_column = positive_column - 1  # shift position
            positive_logits = logits[positive_row, positive_column]
            neutral_row, neutral_column = np.where(inparr == neutral_token)
            neutral_column = neutral_column - 1  # shift position
            neutral_logits = logits[neutral_row, neutral_column]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            positive_labels = labels[np.where(labels.cpu().numpy() == positive_token)]
            loss_positive = loss_fct(positive_logits, positive_labels)
            negative_labels = labels[np.where(labels.cpu().numpy() == negative_token)]
            loss_negative = loss_fct(negative_logits, negative_labels)
            neutral_labels = labels[np.where(labels.cpu().numpy() == neutral_token)]
            loss_neutral = loss_fct(neutral_logits, neutral_labels)
            num_losses = 0
            step_loss_label = 0
            if not loss_positive.isnan():
                step_loss_label += loss_positive
                num_losses += 1
            if not loss_negative.isnan():
                step_loss_label += loss_negative
                num_losses += 1
            if not loss_neutral.isnan():
                step_loss_label += loss_neutral
                num_losses += 1
            step_loss_label /= num_losses
            total_loss_label += step_loss_label

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.model_type == 'gpt2_double':
                    loss_lm = loss_lm.mean()
                    loss_mc = loss_mc.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                if args.model_type == 'gpt2_double':
                    loss_lm = loss_lm / args.gradient_accumulation_steps
                    loss_mc = loss_mc / args.gradient_accumulation_steps

            total_loss += loss.item()
    train_loss = total_loss / steps
    train_loss_label = total_loss_label / steps
    tb_writer.add_scalar("loss", train_loss, 0)
    tb_writer.add_scalar("loss_label", train_loss_label, 0)

    for split in args.eval_splits:
        eval_results = evaluate(args, model, tokenizer, split=split)

        for key, value in eval_results.items():
            tb_writer.add_scalar("metrics/{}/{}".format(split, key), value, 0)

    return


def compute_gpt2_weights_norm(model, results):
    attention_params = []
    all_params = []
    for name, p in model.named_parameters():
        all_params.append(p.data.cpu().numpy())
        if 'bias' not in name and ('c_proj' in name or 'c_attn' in name):
            _, _, block, _, l, _ = name.split('.')
            attention_params.append(p.data.cpu().numpy())
    attention_l2_norms = np.mean([np.linalg.norm(w, ord=2, axis=0).mean() for w in attention_params])
    attention_l1_norms = np.mean([np.linalg.norm(w, ord=1, axis=0).mean() for w in attention_params])
    attention_sum_abs = np.mean([np.linalg.norm(w, ord=1, axis=0).sum() for w in attention_params])
    attention_sum = np.mean([w.sum() for w in attention_params])
    all_l2_norms = np.mean([np.linalg.norm(w, ord=2, axis=0).mean() for w in all_params])
    all_l1_norms = np.mean([np.linalg.norm(w, ord=1, axis=0).mean() for w in all_params])
    all_sum_abs = np.mean([np.linalg.norm(w, ord=1, axis=0).sum() for w in all_params])
    all_sum = np.mean([w.sum() for w in all_params])
    # ipdb.set_trace()
    results['attention_l2norm'] = attention_l2_norms
    results['attention_l1norm'] = attention_l1_norms
    results['attention_sum_abs'] = attention_sum_abs
    results['attention_sum'] = attention_sum
    results['all_l2norm'] = all_l2_norms
    results['all_l1norm'] = all_l1_norms
    results['all_sum_abs'] = all_sum_abs
    results['all_sum'] = all_sum
    return results


def extract_model_weights(model):
    attention_params = []
    all_params = []
    for name, p in model.named_parameters():
        all_params.append(p.data.cpu().numpy())
        if 'bias' not in name and ('c_proj' in name or 'c_attn' in name):
            _, _, block, _, l, _ = name.split('.')
            attention_params.append(p.data.cpu().numpy())

    unfolded_params = []
    for p in all_params:
        unfolded_params.append(p.reshape(-1))
    all_params_unfolded = np.concatenate(unfolded_params, 0)

    unfolded_params = []
    for p in attention_params:
        unfolded_params.append(p.reshape(-1))
    attention_params_unfolded = np.concatenate(unfolded_params, 0)
    return all_params_unfolded, attention_params_unfolded


def extract_gpt2_weights(model):
    params = dict()
    for name, p in model.named_parameters():
        params.setdefault('all', []).append(p.data.cpu().numpy())
        if 'attn' in name:
            params.setdefault('attention', []).append(p.data.cpu().numpy())
        elif 'mlp' in name:
            params.setdefault('feedforward', []).append(p.data.cpu().numpy())
        elif 'ln_' in name:
            params.setdefault('layernorm', []).append(p.data.cpu().numpy())
        # elif 'pooler' in name:
        #     params.setdefault('pooler', []).append(p.data.cpu().numpy())
        elif 'score' in name:
            params.setdefault('classifier', []).append(p.data.cpu().numpy())
        elif 'wte' in name or 'wpe' in name:
            params.setdefault('embeddings', []).append(p.data.cpu().numpy())
        else:
            print(f"unknown name {name}")
            ipdb.set_trace()
    unfolded_params = dict()
    for k in params.keys():
        tmp = []
        for p in params[k]:
            tmp.append(p.reshape(-1))
        unfolded_params[k] = np.concatenate(tmp, 0)

    return unfolded_params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step):
    checkpoint_prefix = "checkpoint"
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    _rotate_checkpoints(args, checkpoint_prefix)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
