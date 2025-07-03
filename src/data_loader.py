import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import BertTokenizer
from utils.tools import load_pickle

from create_dataset import MOSI, MOSEI, PAD, UNK
from data_prepare import *

from transformers import DebertaV2Tokenizer

deberta_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')


class MSADataset(Dataset):
    def __init__(self, config, hp, mode):
        self.config = config
        # Fetch dataset using the new function
        self.data, self.word2id, _ = load_dataset(config, hp, mode)
        self.len = len(self.data)

        config.word2id = self.word2id

    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True, mode=None):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config, hp, mode)
    device=torch.device('cuda')
    
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    
    if mode == 'train':
        hp.n_train = len(dataset)
    elif mode == 'dev':
        hp.n_valid = len(dataset)
    elif mode == 'test':
        hp.n_test = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []
        text_weights = []
        visual_weights = []
        acoustic_weights = []

        for sample in batch:
            if len(sample[0]) > 4: # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            label = sample[1]
            if isinstance(label, np.float32):
                label = torch.IntTensor([label])
            else:
                label = torch.from_numpy(sample[1])
            labels.append(label)
            ids.append(sample[2])
            text_weights.append(sample[0][-3])
            visual_weights.append(sample[0][-2])
            acoustic_weights.append(sample[0][-1])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        # if labels.size(1) == 7:
        #     labels = labels[:,0][:,None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch],target_len=alens.max().item())

        ## BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = 50
        # Create DeBERTa indices using tokenizer

        deberta_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_deberta_sent = deberta_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            deberta_details.append(encoded_deberta_sent)

        # DeBERTa things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in deberta_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in deberta_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in deberta_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids, text_weights, visual_weights, acoustic_weights
    generator = torch.Generator(device=device)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator)

    return data_loader