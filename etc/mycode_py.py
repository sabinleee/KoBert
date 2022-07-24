# import modules
import datetime
import time
import random

# for data analyze and wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from tqdm import tqdm, notebook


# machine learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

# import dataset
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train = pd.read_table("./dataset/ratings_train.txt", usecols = ['document','label'])
test = pd.read_table("./dataset/ratings_test.txt", usecols = ['document','label'])

print(f'numbers of train data: {len(train)}')
print(f'numbers of test data: {len(test)}')
train.head(2)
# train data preprocessing
# CLS : classifier
# SEP : separator
train['document'] = train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train['document'] = train['document'].str.replace('^ +', "")
train['document'].replace('', np.nan, inplace=True)
train = train.dropna(how = 'any')

# before tokenizing
document_bert = ["[CLS] " + str(s) + " [SEP]" for s in train['document']]

# tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
print(f"tokeinzed text : {tokenized_texts[0]}")

# padding
MAX_LEN = max([len(s) for s in tokenized_texts]) + 1
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

# attention mask
attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# split train and validation set   
train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.1)

# attention mask for train and validation set
train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=42, 
                                                       test_size=0.1)

# convert to tensor
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

# data loader
BATCH_SIZE = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
                        train_data, # which is converted to a TensorDataset
                        sampler=train_sampler, # reason of no shuffle is that we need to use validation set
                        batch_size=BATCH_SIZE,
                        pin_memory=True, # GPU memory
                        num_workers=4, # for parallel processing
                        )

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
                        validation_data,
                        sampler=validation_sampler, 
                        batch_size=BATCH_SIZE,
                        pin_memory=True,
                        num_workers=4,
                        )
# test data preprocessing

test['document'] = test['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test['document'] = test['document'].str.replace('^ +', "")
test['document'].replace('', np.nan, inplace=True)
test = test.dropna(how = 'any')

sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in test['document']]
labels = test['label'].values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(
                    test_data, # which is converted to a TensorDataset
                    sampler=test_sampler, # reason of no shuffle is that we need to use validation set
                    batch_size=BATCH_SIZE,
                    pin_memory=True, # GPU memory
                    num_workers=4, # for parallel processing
                    )
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()
# loss function and optimizer
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # Learning rate
                  eps = 1e-8 # Epsilon for AdamW to avoid numerical issues (zero division)
                )

# number of training epochs
epochs = 5

# loss function to calculate loss
loss_fn = nn.CrossEntropyLoss()

# total number of training step 
# len(train_dataloader) : number of batches
total_steps = len(train_dataloader) * epochs

warmup_ratio = 0.1
warmup_step = int(total_steps * warmup_ratio)

# scheduler to decrease learning rate
scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_step,
                                            num_training_steps = total_steps)
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# time format
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # change to hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# zero gradients
model.zero_grad()

# loop over epochs
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # set time recorder
    t0 = time.time()

    # intialize the loss
    total_loss = 0

    # set model to train mode
    model.train()
        
    # get step and batch from train_dataloader
    for step, batch in enumerate(train_dataloader):
        # progress bar
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # unpack the inputs from dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass                
        # **batch
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        # get loss
        loss = outputs[0]

        # get the total loss
        total_loss += loss.item()

        # Backward pass, gradient calculation
        loss.backward()

        # gradient clipping to avoid gradient exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update weight parameters from backpropagation
        optimizer.step()

        # learning rate decay by scheduler
        scheduler.step()

        # zero gradients
        model.zero_grad()
        
    # evaluate the average loss over the epoch
    avg_train_loss = total_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # set time recorder
    t0 = time.time()

    # set model to eval mode
    model.eval()

    # initialize the loss and accuracy
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # get batch from val_dataloader
    for batch in validation_dataloader:
        # batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # unpack the inputs from dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # no need to calculate gradients because of eval mode, so set requires_grad to False
        with torch.no_grad():     
            # Forward pass
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # get loss
        logits = outputs[0]

        # loss to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # calculate evaluation accuracy
        # flat_accuracy : calculate accuracy of each batch
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")
t0 = time.time()

# 평가모드로 변경
model.eval()

# 변수 초기화
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 데이터로더에서 배치만큼 반복하여 가져옴
for step, batch in enumerate(test_dataloader):
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # 배치를 GPU에 넣음
    batch = tuple(t.to(device) for t in batch)
    
    # 배치에서 데이터 추출
    b_input_ids, b_input_mask, b_labels = batch
    
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))
bert_base_multilingual_cased_accuracy = eval_accuracy/nb_eval_steps
## KoBERT from SKT

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
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

dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("ratings_test.txt", field_indices=[1,2], num_discard_samples=1)
# max_len = 64
# batch_size = 32
# warmup_ratio = 0.1
# num_epochs = 5
# max_grad_norm = 1
# log_interval = 200
# learning_rate =  5e-5

tok = tokenizer.tokenize

data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, 64, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, 64, True, False)

train_dataloader = DataLoader(
                    data_train, 
                    batch_size=BATCH_SIZE, 
                    num_workers=4,
                    shuffle=True,
                    pin_memory=True
                    )
test_dataloader = DataLoader(
                    data_test, 
                    batch_size=BATCH_SIZE, 
                    num_workers=4,
                    shuffle=True,
                    pin_memory=True
                    )
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
    
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
# max_len = 64
# batch_size = 32
# warmup_ratio = 0.1
# num_epochs = 5
# max_grad_norm = 1
# log_interval = 200
# learning_rate =  5e-5



no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # Learning rate
                  eps = 1e-8 # Epsilon for AdamW to avoid numerical issues (zero division)
                )

loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * epochs
warmup_step = int(t_total * 0.1)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
for e in range(epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(notebook.tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % 200 == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(notebook.tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    kobert_accuracy = test_acc / (batch_id+1)
# text classification
def predict_input_text(text):
    transform = nlp.data.BERTSentenceTransform(
                tokenizer.tokenize, max_seq_length=128, vocab=vocab, pad=True, pair=False)

    sentence = text
    sentence = transform([sentence])

    sentence_dataloader = DataLoader(sentence, batch_size=1, shuffle=False)

    token_ids, valid_length, segment_ids = sentence_dataloader
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length

    out = model(token_ids, valid_length, segment_ids)
    predict = torch.max(out, 1)[1].item()
    if predict == 0:
        return "부정적인 문장입니다"
    else:
        return "긍정적인 문장입니다"
    
print(predict_input_text('이건 좀... 아니지 않니..?'))