
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_recall_fscore_support, matthews_corrcoef
from scipy.stats.mstats import gmean
import ipdb
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def compute_single_term_polarity(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|term|>')[0].strip()
        aspect_term = ' '.join(tokenizer.decode(inp).split('<|term|>')[-1].split('<|endofterm|>')[0].strip().split(' ')[:-1])
        inp_dec = f"{inp_text} <|term|> {aspect_term}"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()
            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        # only consider first generated token
        gen_term_polarity = gen_text.split('<|term|>')[-1].split('<|endofterm|>')[0].split(',')[0]
        review = gen_text.split('<|term|>')[0].strip()
        gen_text = f"{review} <|term|> {gen_term_polarity} <|endofterm|> <|endoftext|>"

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_sentiment(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|sentiment|>')[0].strip()
        inp_dec = f"{inp_text} <|sentiment|>"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()
            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break

        gen_text = tokenizer.decode(indexed_tokens)

        # only consider first generated token
        gen_sentiment = gen_text.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()
        review = gen_text.split('<|sentiment|>')[0].strip()
        gen_text = f"{review} <|sentiment|> {gen_sentiment} <|endofsentiment|> <|endoftext|>"

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_sst2(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|sentiment|>')[0].strip()
        inp_dec = f"{inp_text} <|sentiment|>"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()
            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break

        gen_text = tokenizer.decode(indexed_tokens)

        # only consider first generated token
        gen_sentiment = gen_text.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()
        review = gen_text.split('<|sentiment|>')[0].strip()
        gen_text = f"{review} <|sentiment|> {gen_sentiment} <|endofsentiment|> <|endoftext|>"

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_single_category_polarity(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|category|>')[0].strip()
        aspect_category = ' '.join(tokenizer.decode(inp).split('<|category|>')[-1].split('<|endofcategory|>')[0].strip().split(' ')[:-1])
        inp_dec = f"{inp_text} <|category|> {aspect_category}"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()

            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        # only consider first generated token
        gen_category_polarity = gen_text.split('<|category|>')[-1].split('<|endofcategory|>')[0].split(',')[0]
        review = gen_text.split('<|category|>')[0].strip()
        gen_text = f"{review} <|category|> {gen_category_polarity} <|endofcategory|> <|endoftext|>"


        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_aspect_term(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|term|>')[0].strip()
        inp_dec = f"{inp_text} <|term|>"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()

            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_term_polarity(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|term|>')[0].strip()
        aspect_term = ' '.join(tokenizer.decode(inp).split('<|term|>')[-1].split('<|endofterm|>')[0].strip().split(' ')[:-1])
        inp_dec = f"{inp_text} <|term|> {aspect_term}"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()
            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_category_polarity(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|category|>')[0].strip()
        aspect_category = ' '.join(tokenizer.decode(inp).split('<|category|>')[-1].split('<|endofcategory|>')[0].strip().split(' ')[:-1])
        inp_dec = f"{inp_text} <|category|> {aspect_category}"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()
            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_aspect_category(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|category|>')[0].strip()
        inp_dec = f"{inp_text} <|category|>"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()

            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_aspect_term_aspect_category(model, input, label, tokenizer, args):

    break_tokens = tokenizer.encode(tokenizer._eos_token.content)
    MAX_LEN = args.block_size
    batch_pred = []
    batch_ground = []
    for inp, ground in zip(input, label):
        inp_text = tokenizer.decode(inp).split('<|term|>')[0].strip()
        inp_dec = f"{inp_text} <|term|>"
        ground_dec = tokenizer.decode(ground)

        indexed_tokens = tokenizer.encode(inp_dec)
        tokens_tensor = tokenizer.encode(inp_dec, return_tensors='pt').to(args.device)
        predicted_index = indexed_tokens[-1]
        while predicted_index not in break_tokens:
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            probs, indexes = torch.max(torch.softmax(predictions, -1), -1)

            predicted_index = indexes[0, -1].item()

            indexed_tokens += [predicted_index]

            tokens_tensor = torch.tensor([indexed_tokens]).to(args.device)
            if len(indexed_tokens) > MAX_LEN:
                break
        gen_text = tokenizer.decode(indexed_tokens)

        batch_pred.append(gen_text)
        batch_ground.append(ground_dec)

    return batch_pred, batch_ground


def compute_oos_metrics(result_dict, labels, predictions):

    in_true = []
    in_pred = []
    in_correct = 0
    in_total = 0

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    oos_true = []
    oos_pred = []

    for true, pred in zip(labels, predictions):
        l = true.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()
        p = pred.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

        ####### for Acc(in) metric #############
        if l != 'out of scope':

            in_true.append(l)
            in_pred.append(p)

            in_total += 1
            if l == p:
                in_correct += 1

        ####### for Acc(out) metric #############
        if l == 'out of scope':
            oos_true.append(1)
        else:
            oos_true.append(0)

        if p == 'out of scope':
            oos_pred.append(1)
        else:
            oos_pred.append(0)

    oos_acc = accuracy_score(oos_true, oos_pred)
    oos_recall = recall_score(oos_true, oos_pred)

    in_acc = in_correct / in_total
    all_acc = all_correct / all_total

    result_dict.update(
        {
            'acc_in': in_acc,
            'acc_full': all_acc,
            'acc_oos': oos_acc,
            'recall_oos': oos_recall
        }
    )

    return result_dict


def compute_oos_metrics_hulu(result_dict, labels, predictions, probabilites, label_set, args):

    in_true = []
    in_pred = []
    in_correct = 0
    in_total = 0

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    oos_true = []
    oos_pred = []

    threshold = args.confidence_threshold


    for true, pred, prob in zip(labels, predictions, probabilites):
        l = true.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()
        p = pred.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()

        if p not in label_set or prob < threshold:
            p = 'out of scope'


        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

        ####### for Acc(in) metric #############
        if l != 'out of scope':

            in_true.append(l)
            in_pred.append(p)

            in_total += 1
            if l == p:
                in_correct += 1

        ####### for Acc(out) metric #############
        if l == 'out of scope':
            oos_true.append(1)
        else:
            oos_true.append(0)

        if p == 'out of scope':
            oos_pred.append(1)
        else:
            oos_pred.append(0)

    oos_acc = accuracy_score(oos_true, oos_pred)
    oos_prec, oos_recall, oos_fscore, _ = precision_recall_fscore_support(oos_true, oos_pred, labels=[1])
    in_acc = in_correct / in_total
    all_acc = all_correct / all_total


    in_true_noconfidence = []
    in_pred_noconfidence = []
    in_correct_noconfidence = 0
    in_total_noconfidence = 0
    for true, pred, prob in zip(labels, predictions, probabilites):
        l = true.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()
        p = pred.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()


        ####### for Acc(in) metric #############
        if l != 'out of scope':

            in_true_noconfidence.append(l)
            in_pred_noconfidence.append(p)

            in_total_noconfidence += 1
            if l == p:
                in_correct_noconfidence += 1

    in_acc_noconfidence = in_correct_noconfidence / in_total_noconfidence

    result_dict.update(
        {
            'acc_in': in_acc,
            'acc_in_noconfidence': in_acc_noconfidence,
            'acc_full': all_acc,
            'acc_oos': oos_acc,
            'oos_precision': oos_prec[0],
            'oos_recall': oos_recall[0],
            'oos_fscore': oos_fscore[0],
        }
    )

    return result_dict


def compute_metrics(result_dict, labels, predictions, task_labels, args):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()
        p = pred.split('<|intent|>')[-1].split('<|endofintent|>')[0].strip()

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

    all_acc = all_correct / all_total

    result_dict.update(
        {
            'acc': all_acc,
        }
    )

    return result_dict


def compute_term_polarity_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|term|>')[-1].split('<|endofterm|>')[0].strip().split(' ')[-1]
        p = pred.split('<|term|>')[-1].split('<|endofterm|>')[0].strip().split(' ')[-1]

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    result_dict.update(
        {
            'term_polarity_acc': all_acc,
            'term_polarity_prec': prec,
            'term_polarity_recall': recall,
            'term_polarity_fscorce': fscore,
        }
    )

    return result_dict


def compute_sentiment_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()
        p = pred.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    result_dict.update(
        {
            'sentiment_acc': all_acc,
            'sentiment_prec': prec,
            'sentiment_recall': recall,
            'sentiment_fscorce': fscore,
        }
    )

    return result_dict


def compute_sst2_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()
        p = pred.split('<|sentiment|>')[-1].split('<|endofsentiment|>')[0].strip()

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    result_dict.update(
        {
            'sentiment_acc': all_acc,
            'sentiment_prec': prec,
            'sentiment_recall': recall,
            'sentiment_fscorce': fscore,
        }
    )

    return result_dict


def compute_category_polarity_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip().split(' ')[-1]
        p = pred.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip().split(' ')[-1]

        ####### for Acc(full) metric #############
        all_true.append(l)
        all_pred.append(p)

        all_total += 1
        if l == p:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    result_dict.update(
        {
            'category_polarity_acc': all_acc,
            'category_polarity_prec': prec,
            'category_polarity_recall': recall,
            'category_polarity_fscorce': fscore,
        }
    )

    return result_dict


def sort_output(ground, pred):
    ground_dic = []
    for t in ground.strip().split(','):
        if t in ['', ' ']:
            continue
        term = ' '.join(t.split(' ')[:-1]).strip()
        label = t.split(' ')[-1].strip()
        if (term, label) not in ground_dic:
            ground_dic.append((term, label))

    pred_dic = []
    for t in pred.strip().split(','):
        if t in ['', ' ']:
            continue
        term = ' '.join(t.split(' ')[:-1]).strip()
        label = t.split(' ')[-1].strip()
        if (term, label) not in pred_dic:
            pred_dic.append((term, label))

    ground_dic = sorted(ground_dic, key=lambda a: a[0])
    pred_dic = sorted(pred_dic, key=lambda a: a[0])

    ground_list = []
    for term, label in ground_dic:
        ground_list.append(f"{term} {label}")
    ground_text = ', '.join(ground_list)

    pred_list = []
    for term, label in pred_dic:
        pred_list.append(f"{term} {label}")
    pred_text = ', '.join(pred_list)

    return ground_text, pred_text


# Aspect Extraction (no offsets considered)
def aspect_extraction(correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(len(correct)):
        cor = [t.split(' ')[:-1] for t in correct[i].split(',')]
        pre = [t.split(' ')[:-1] for t in predicted[i].split(',')]
        common += len([a for a in pre if a in cor])
        retrieved += len(pre)
        relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
    return p, r, f1, common, retrieved, relevant


# Aspect Category Detection
def category_detection(correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(len(correct)):
        cor = [t.split(' ')[:-1] for t in correct[i].split(',')]
        pre = [t.split(' ')[:-1] for t in predicted[i].split(',')]
        common += len([c for c in pre if c in cor])
        retrieved += len(pre)
        relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (1 + b ** 2) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
    return p, r, f1, common, retrieved, relevant


def aspect_polarity_estimation(correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    true_relevant = 0.
    false_positive = 0.
    false_negative = 0.
    for i in range(len(correct)):
        cor = correct[i].split(',')
        pre = predicted[i].split(',')
        common += len([a for a in pre if a in cor])
        false_positive += len([a for a in pre if a not in cor])
        false_negative += len([a for a in cor if a not in pre])

        retrieved += len(pre)
        true_relevant += len(cor)

    acc = common / retrieved
    acc_true = common / true_relevant
    acc_correct = common / (false_positive + false_negative + common)
    return acc, acc_true, acc_correct, common, retrieved


def aspect_category_polarity_estimation(correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    true_relevant = 0.
    false_positive = 0.
    false_negative = 0.
    for i in range(len(correct)):
        cor = correct[i].split(',')
        pre = predicted[i].split(',')
        common += len([a for a in pre if a in cor])
        false_positive += len([a for a in pre if a not in cor])
        false_negative += len([a for a in cor if a not in pre])
        retrieved += len(pre)
        true_relevant += len(cor)
    acc = common / retrieved
    acc_true = common / true_relevant
    acc_correct = common / (false_positive + false_negative + common)
    return acc, acc_true, acc_correct, common, retrieved


def compute_aspect_term_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()
        p = pred.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()

        l_new, p_new = sort_output(l, p)

        ####### for Acc(full) metric #############
        all_true.append(l_new)
        all_pred.append(p_new)

        all_total += 1
        if l_new == p_new:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    # compute aspect term extraction
    prec_extract, recall_extract, f1_extract, _, _, _ = aspect_extraction(all_true, all_pred)

    # compute aspect term polarity
    acc_polarity, acc_polarity_true, acc_polarity_correct, _, _ = aspect_polarity_estimation(all_true, all_pred)

    result_dict.update(
        {
            'term_acc': all_acc,
            'term_prec': prec,
            'term_recall': recall,
            'term_fscorce': fscore,

            'term_extract_prec': prec_extract,
            'term_extract_recall': recall_extract,
            'term_extract_fscorce': f1_extract,

            'term_polarity_acc': acc_polarity,
            'term_polarity_acc_true': acc_polarity_true,
            'term_polarity_acc_correct': acc_polarity_correct
        }
    )

    return result_dict


def compute_aspect_category_metrics(result_dict, labels, predictions):

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l = true.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()
        p = pred.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()

        l_new, p_new = sort_output(l, p)

        ####### for Acc(full) metric #############
        all_true.append(l_new)
        all_pred.append(p_new)

        all_total += 1
        if l_new == p_new:
            all_correct += 1

    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    # compute aspect category detection
    prec_detect, recall_detect, f1_detect, _, _, _ = category_detection(all_true, all_pred)

    # compute aspect category polarity
    acc_polarity, acc_polarity_true, acc_polarity_correct, _, _ = aspect_category_polarity_estimation(all_true, all_pred)

    result_dict.update(
        {
            'category_acc': all_acc,
            'category_prec': prec,
            'category_recall': recall,
            'category_fscorce': fscore,

            'category_detection_prec': prec_detect,
            'category_detection_recall': recall_detect,
            'category_detection_fscorce': f1_detect,

            'category_polarity_acc': acc_polarity,
            'category_polarity_acc_true': acc_polarity_true,
            'category_polarity_acc_correct': acc_polarity_correct
        }
    )

    return result_dict


def compute_aspect_term_aspect_category_metrics(result_dict, labels, predictions):

    all_term_true = []
    all_term_pred = []
    all_term_correct = 0
    all_term_total = 0

    all_category_true = []
    all_category_pred = []
    all_category_correct = 0
    all_category_total = 0

    all_true = []
    all_pred = []
    all_correct = 0
    all_total = 0

    for true, pred in zip(labels, predictions):
        l_term = true.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()
        p_term = pred.split('<|term|>')[-1].split('<|endofterm|>')[0].strip()

        l_category = true.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()
        p_category = pred.split('<|category|>')[-1].split('<|endofcategory|>')[0].strip()


        l_term_sorted, p_term_sorted = sort_output(l_term, p_term)
        l_category_sorted, p_category_sorted = sort_output(l_category, p_category)

        l_joint = f"<|term|> {l_term_sorted} <|endofterm|> <|category|> {l_category_sorted} <|endofcategory|>"
        p_joint = f"<|term|> {p_term_sorted} <|endofterm|> <|category|> {p_category_sorted} <|endofcategory|>"

        ####### Term Acc(full) metric #############
        all_term_true.append(l_term_sorted)
        all_term_pred.append(p_term_sorted)

        all_term_total += 1
        if l_term_sorted == p_term_sorted:
            all_term_correct += 1

        ####### Category Acc(full) metric #############
        all_category_true.append(l_category_sorted)
        all_category_pred.append(p_category_sorted)

        all_category_total += 1
        if l_category_sorted == p_category_sorted:
            all_category_correct += 1

        ####### Joint Term + Category Acc(full) metric #############
        all_true.append(l_joint)
        all_pred.append(p_joint)

        all_total += 1
        if l_joint == p_joint:
            all_correct += 1

    # term metric #
    acc_term = accuracy_score(all_term_true, all_term_pred)
    prec_term, recall_term, fscore_term, _ = precision_recall_fscore_support(all_term_true, all_term_pred, average='macro')

    # category metric #
    acc_category = accuracy_score(all_category_true, all_category_pred)
    prec_category, recall_category, fscore_category, _ = precision_recall_fscore_support(all_category_true, all_category_pred, average='macro')

    # joint metrics #
    all_acc = all_correct / all_total
    acc = accuracy_score(all_true, all_pred)
    assert acc == all_acc
    prec, recall, fscore, _ = precision_recall_fscore_support(all_true, all_pred, average='macro')

    # compute aspect term extraction
    prec_term_extract, recall_term_extract, f1_term_extract, _, _, _ = aspect_extraction(all_term_true, all_term_pred)

    # compute aspect term polarity
    acc_term_polarity, acc_term_polarity_true, acc_term_polarity_correct, _, _ = aspect_polarity_estimation(all_term_true, all_term_pred)

    # compute aspect category detection
    prec_category_detect, recall_category_detect, f1_category_detect, _, _, _ = category_detection(all_category_true, all_category_pred)

    # compute aspect category polarity
    acc_category_polarity, acc_category_polarity_true, acc_category_polarity_correct, _, _ = aspect_category_polarity_estimation(all_category_true,
                                                                                                      all_category_pred)

    result_dict.update(
        {
            'joint_acc': all_acc,
            'joint_prec': prec,
            'joint_recall': recall,
            'joint_fscorce': fscore,

            'term_acc': acc_term,
            'term_prec': prec_term,
            'term_recall': recall_term,
            'term_fscorce': fscore_term,

            'term_extract_prec': prec_term_extract,
            'term_extract_recall': recall_term_extract,
            'term_extract_fscorce': f1_term_extract,

            'term_polarity_acc': acc_term_polarity,
            'term_polarity_acc_true': acc_term_polarity_true,
            'term_polarity_acc_correct': acc_term_polarity_correct,


            'category_acc': acc_category,
            'category_prec': prec_category,
            'category_recall': recall_category,
            'category_fscorce': fscore_category,

            'category_detection_prec': prec_category_detect,
            'category_detection_recall': recall_category_detect,
            'category_detection_fscorce': f1_category_detect,

            'category_polarity_acc': acc_category_polarity,
            'category_polarity_acc_true': acc_category_polarity_true,
            'category_polarity_acc_correct': acc_category_polarity_correct
        }
    )

    return result_dict

