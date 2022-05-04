
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
import os
import logging
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset_gpt2_double(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            raw = f.read().splitlines()
            lines = [line.split('\t')[0] for line in raw if (len(line) > 0 and not line.isspace())]
            labels = [[int(line.split('\t')[1])] for line in raw if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False, split='val'):
    if not evaluate:
        file_path = args.train_data_file
    else:
        if split == 'val':
            file_path = args.eval_data_file
        elif split == 'test':
            file_path = args.test_data_file
        else:
            raise TypeError('split value is unknown')
    # file_path = args.eval_data_file if evaluate else args.train_data_file
    
    if not evaluate:
        if args.model_type == 'gpt2_double':
            return LineByLineTextDataset_gpt2_double(tokenizer, args, file_path=file_path, block_size=args.block_size)
        else:
            return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        if args.model_type == 'gpt2_double':
            return LineByLineTextDataset_gpt2_double(tokenizer, args, file_path=file_path, block_size=args.block_size)
        else:
            return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def get_dataloader(dataset, tokenizer, args, split='train'):

    def collate(examples):
        if tokenizer._pad_token is None:
            if args.model_type == 'gpt2_double':
                text = [ex[0] for ex in examples]
                labels = [ex[1] for ex in examples]
                padded_labels = torch.stack(labels, 0)
                return pad_sequence(text, batch_first=True), padded_labels
            else:
                return pad_sequence(examples, batch_first=True)
        if args.model_type == 'gpt2_double':
            text = [ex[0] for ex in examples]
            labels = [ex[1] for ex in examples]
            padded_labels = torch.stack(labels, 0)
            return pad_sequence(text, batch_first=True, padding_value=tokenizer.pad_token_id), padded_labels
        else:
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)

    return dataloader, args
