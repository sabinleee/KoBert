# import modules
import datetime
import time
import logging
import random

# for data analyze and wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import urllib.request
import easydict
from tqdm import tqdm, notebook

# machine learning
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertModel
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
import gluonnlp as nlp

# kobert
from kobert_tokenizer import KoBERTTokenizer

def preprocessing(df):
    logging.info(f"Before preprocessing df's shape: {df.shape}")
    
    df['document'] = df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    df['document'] = df['document'].str.replace('^ +', "")
    df['document'] = df['document'].replace('', np.nan)
    df = df.dropna(how = 'any')
    
    logging.info(f"After preprocessing df's shape: {df.shape}")
    return df    

#TODO easydict 형식으로 parameters 정리 가능?
class ModelInit():
    def __init__(self, model_name):
        self.args = easydict.EasyDict({'bert_model': model_name,
                                        'n_class': 2, 'max_token_len': 512})
        
        self.tokenizer = None
        self.vocab = None
        self.model = None
        if model_name == "bert-base-multilingual-cased":
            self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
            self.model = BertForSequenceClassification.from_pretrained(self.args.bert_model, num_labels=self.args.n_class, return_dict=False)

        elif model_name == "skt/kobert-base-v1":
            self.tokenizer = KoBERTTokenizer.from_pretrained(self.args.bert_model)
            self.model = BertModel.from_pretrained(self.args.bert_model, num_labels=self.args.n_class, return_dict=False)
            self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(self.tokenizer.vocab_file, padding_token='[PAD]')
        
        self.train_batch_size = 32
        self.test_batch_size = 16
        self.no_decay = None
        self.optimizer_grouped_parameters = None
        self.optimizer = None
        self.loss_fn = None
        self.t_total = None
        self.warmup_step = None
        self.scheduler = None
        self.train_loss_per_epoch = []
        self.validation_accuracy_per_epoch = []
        self.accuracy = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.epochs = None
    
    def define_hyperparameters(self, model, train_dataloader, epochs):
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(model.parameters(),
                        lr = 5e-5, # Learning rate
                        eps = 1e-8 # Epsilon for AdamW to avoid numerical issues (zero division)
                        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.t_total = len(train_dataloader) * epochs
        self.warmup_step = int(self.t_total * 0.1)

        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_step, num_training_steps=self.t_total)
        self.epochs = epochs

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
