"""
@author: Vrijesh Kunwar
"""


import numpy as np
import wandb
import warnings
import argparse

import math
import time
import random
from typing import Tuple


import torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
from collections import Counter
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Set random seed for reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
import os

wandb.login(key='3c21150eb43b007ee446a1ff6e87f640ec7528c4')


# Define the function for loading data
def dt_ld(path):
    # Load data from the CSV file using pandas
    dt = pd.read_csv(path, sep=',', header=None, names=["eng", "hin", ""], skip_blank_lines=True, index_col=None, encoding='utf-8')

    # Filter out rows where either 'eng' or 'hin' column is NaN
    word_data = dt.dropna(subset=['eng', 'hin'])

    # Select only 'eng' and 'hin' columns
    word_data = word_data[['eng', 'hin']]

    return word_data

# Define the file paths for train, valid, and test data on your local machine
train_file_path = r"C:\Users\ASL 5\Downloads\aksharantar_sampled\hin\hin_train.csv"
valid_file_path = r"C:\Users\ASL 5\Downloads\aksharantar_sampled\hin\hin_valid.csv"
test_file_path = r"C:\Users\ASL 5\Downloads\aksharantar_sampled\hin\hin_test.csv"

# Load the data using the defined function
train_data = dt_ld(train_file_path)
valid_data = dt_ld(valid_file_path)
test_data = dt_ld(test_file_path)

# Extracting 'eng' and 'hin' columns into lists
test_hin = list(test_data['hin'])
test_eng = list(test_data['eng'])

# Creating tokenizied dataset here

# Define a function to generate unique tokens for Hindi and English words from a dataset
def uni_tok(dataset):
  # Extract Hindi and English sentences from the dataset
  hin = dataset['hin'].values
  eng = dataset['eng'].values

  # Initialize sets to store unique tokens for Hindi and English
  hin_token = set()
  eng_token = set()

  # Iterate through pairs of Hindi and English sentences
  for i, j in zip(eng, hin):
    # Iterate through characters in Hindi sentence
    for chare in j:
      # Add each character to Hindi tokens set
      hin_token.add(chare)
    # Iterate through characters in English sentence
    for chare in i:
      # Add each character to English tokens set
      eng_token.add(chare)

  # Sort the sets and convert them to lists
  hin_token = sorted(list(hin_token))
  eng_token = sorted(list(eng_token))

  # Return the lists of unique Hindi and English tokens
  return hin_token, eng_token

# Call the function to generate tokens for the training data
hindi_token, english_token = uni_tok(train_data)

# Creating token map here
def tokenize_map(language_tokens , english_tokens):

    # Create English token map with characters as keys and their index + 1 as values
    english_token_map = dict([(ch,i+1) for i,ch in enumerate(english_tokens)])

    # Create language token map with characters as keys and their index + 1 as values
    language_token_map = dict([(ch,i+1) for i,ch in enumerate(language_tokens)])

    # Create a reverse language token map with index + 1 as keys and characters as values
    reverse_language_token_map = dict([(i+1,ch) for i,ch in enumerate(language_tokens)])

    # Adding blank space to both token maps with value 0
    language_token_map[" "] = 0
    english_token_map[" "] = 0

    # Add special tokens for beginning and end of sentence in language token map
    language_token_map[';']=65
    language_token_map['.']=66

    # Add special tokens for beginning and end of sentence in English token map
    english_token_map[';']=27
    english_token_map['.']=28

    # Add <unk> token for unknown characters to language token maps
    language_token_map['<unk>']=64

    # Update the reverse language token map with special tokens
    reverse_language_token_map[64]='<unk>'
    reverse_language_token_map[65]=';'
    reverse_language_token_map[66]='.'
    reverse_language_token_map[0]=''

    # Return the Marathi token map, English token map, and reverse Marathi token map
    return language_token_map, reverse_language_token_map, english_token_map

# Call the function to generate token maps for hindi and english
hin_token_map, reverse_hin_token_map, eng_token_map = tokenize_map(hindi_token, english_token)


# Assigning values from the 'hin' and 'eng' columns of the test_data DataFrame to variables 'hiii' and 'ennn' respectively.
hiii = test_data['hin'].values
ennn = test_data['eng'].values

# Adding ';' at the beginning and '.' at the end of each element in the 'hiii' and 'ennn' arrays.
# Adding special characters like ';' and '.' the start and end of a sentence or phrase, which is necessary for subsequent processing or analysis being performed on the strings.
hiii = ';' + hiii + '.'
ennn = ';' + ennn + '.'

# Finding the maximum length of a string in the 'hiii' array.
maximum_hin = max([len(i) for i in hiii])
# Finding the maximum length of a string in the 'ennn' array.
maximum_eng = max([len(i) for i in ennn])


# NOw we do one hot encoding
#unknown token present in validation set as 'r.(in hindi)'
unknown_token=64
def process(data):
    x,y = data['eng'].values, data['hin'].values
    x = ";" + x + "."
    y = ";" + y + "."
    
    a = torch.zeros((len(x),maximum_eng),dtype=torch.int64)
    
    b = torch.zeros((len(y),maximum_eng),dtype=torch.int64)

    data=[]
    for i,(xx,yy) in enumerate(zip(x,y)):
        for j,ch in enumerate(xx):
            a[i,j] = eng_token_map[ch]

        #a[i,j+1:] = eng_token_map[" "]
        for j,ch in enumerate(yy):
            if ch in hin_token_map:
             b[i,j] = hin_token_map[ch]
            else:
              b[i,j]= unknown_token

    data = [(a[i], b[i]) for i in range(len(x))]
    return data

train_procs = process(train_data)
valid_procs = process(valid_data)
test_procs = process(test_data)

# For reading the words, bro...

def reverse_tokenize(data):
    # Convert each element in data to an integer
    data = map(int, data)
    # Map each integer to its corresponding character using reverse_marathi_token_map
    characters = map(reverse_hin_token_map.get, data)
    # Join the characters into a single string
    predicted_seq = ''.join(characters)
    return predicted_seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_bitch_SIZE = 16

PAD_IDX = 0
BOS_IDX = ';'
EOS_IDX = '.'



train_itersn = DataLoader(train_procs, batch_size=BATCH_bitch_SIZE,
                        shuffle=False)
valid_itersn = DataLoader(valid_procs, batch_size=BATCH_bitch_SIZE,
                        shuffle=False)
test_itersn = DataLoader(test_procs, batch_size=BATCH_bitch_SIZE,
                       shuffle=False)


# Attension implimenststsiomn

class Encoder(nn.Module):
    def __init__(self, input_dimension: int, emb_dimension: int, enc_hid_dimension: int, dec_hid_dimension: int, dropout: float):
        super().__init__()

        self.input_dimension = input_dimension
        self.emb_dimension = emb_dimension
        self.enc_hid_dimension = enc_hid_dimension
        self.dec_hid_dimension = dec_hid_dimension
        self.dropout = dropout

        self.embedding = nn.Embedding(self.input_dimension, self.emb_dimension)


        self.rnn = nn.GRU(emb_dimension, enc_hid_dimension, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dimension * 2, dec_hid_dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        src = src.permute(1,0)

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dimension: int, dec_hid_dimension: int, attn_dimension: int):
        super().__init__()

        self.enc_hid_dimension = enc_hid_dimension
        self.dec_hid_dimension = dec_hid_dimension

        self.attn_in = (enc_hid_dimension * 2) + dec_hid_dimension

        self.attn = nn.Linear(self.attn_in, attn_dimension)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dimension: int, emb_dimension: int, enc_hid_dimension: int, dec_hid_dimension: int, dropout: int, attention: nn.Module):
        super().__init__()

        self.emb_dimension = emb_dimension
        self.enc_hid_dimension = enc_hid_dimension
        self.dec_hid_dimension = dec_hid_dimension
        self.output_dimension = output_dimension
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dimension, emb_dimension)

        self.rnn = nn.GRU((enc_hid_dimension * 2) + emb_dimension, dec_hid_dimension)

        self.out = nn.Linear(self.attention.attn_in + emb_dimension, output_dimension)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self, input_: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor]:

        input_ = input_.unsqueeze(0)
        input_ = input_.permute(1,0)
        embedded = self.dropout(self.embedding(input_))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        embedded = embedded.permute(1,0,2)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dimension
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        #ATTENTION HEAT MAP METHOD
 
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        trg = trg.permute(1,0)
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs








if __name__ == '__main__':    
  
    parser = argparse.ArgumentParser(description='Impliment Seqence to sequence model RNN, LSTM, GRU andapplu Attention in it')
    parser.add_argument('--wandb_entity', type=str, default='ed23d015', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='DL_Assignment_3', help='Name of the wandb project')
    parser.add_argument('--embed_dimension', type=int, default=128, help='Size of embedding')
    parser.add_argument('--hidden_layer_dimension', type=int, default=512, help='Size of Hidden layer for the run')
    parser.add_argument('--attention_dimension', type=int, default=64, help='attention_dimension to optimize model')
    parser.add_argument('--dropout', type=int, default=0.3, help='choices:[0.3,0.5,0.6]')
    

    args = parser.parse_args()

    #train_images,val_images,train_labels ,val_labels = dataset_preprocess(args.dataset)
    
    INPUT_DIMENSION = len(eng_token_map)
    OUTPUT_DIMENSION = len(hin_token_map)
    ENC_EMB_DIMENSION = 512
    DEC_EMB_DIMENSION = 512
    ENC_HID_DIMENSION  = 256
    DEC_HID_DIMENSION  = 256
    ATTN_DIMENSION  = 256
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    encoder_bkl = Encoder(INPUT_DIMENSION , ENC_EMB_DIMENSION, ENC_HID_DIMENSION, DEC_HID_DIMENSION, ENC_DROPOUT)
    attension_bkl = Attention(ENC_HID_DIMENSION, DEC_HID_DIMENSION, ATTN_DIMENSION)
    decoder_bkl = Decoder(OUTPUT_DIMENSION , DEC_EMB_DIMENSION,  ENC_HID_DIMENSION, DEC_HID_DIMENSION, DEC_DROPOUT, attension_bkl)
    model_bkl = Seq2Seq(encoder_bkl, decoder_bkl, device).to(device)
    
    
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    
    #added custom weights
    model_bkl.apply(init_weights)
    
    optimizer = optim.Adam(model_bkl.parameters())
    
    
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
          'name': 'valid_accuracy',
          'goal': 'maximize'
        },
        'parameters': {
            'embed_dimension': {
                'values': [args.embed_dimension]
            },
            'hidden_layer_dimension': {
                'values': [args.hidden_layer_dimension]
            },
            'attention_dimension':{
                'values':[args.attention_dimension]
            },
            'dropout':{
                'values':[args.dropout]
            },
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, entity='ed23d015', project="DL_Assignment_3")
    
    def sweep_train():
      # Default values for hyper-parameters we're going to sweep over
      config_defaults = {
          'embed_dimension':256,
          'hidden_layer_dimension':256,
          'attention_dimension':128,
          'dropout':0.6,
      }
    
      # Initialize a new wandb run
      wandb.init(project='DL_Assignment_3', entity='ed23d015',config=config_defaults)
      wandb.run.name = 'embed_dimension:'+ str(wandb.config.embed_dimension)+' ;hl:'+str(wandb.config.hidden_layer_dimension)+ ' ;attention_dimension:'+str(wandb.config.attention_dimension)+ ' ;dropout:'+str(wandb.config.dropout)
    
      config = wandb.config
      embed_dimension = config.embed_dimension
      hidden_layer_dimension = config.hidden_layer_dimension
      attention_dimension = config.attention_dimension
      dropout = config.dropout
    
      # Doing Model training here
      INPUT_DIM = len(eng_token_map)
      OUTPUT_DIM = len(hin_token_map)
    
    
      encoder_bkl = Encoder(INPUT_DIM, embed_dimension, hidden_layer_dimension, hidden_layer_dimension, dropout)
    
      attension_bkl = Attention(hidden_layer_dimension, hidden_layer_dimension, attention_dimension)
    
      decoder_bkl = Decoder(OUTPUT_DIM, embed_dimension, hidden_layer_dimension, hidden_layer_dimension, dropout, attension_bkl)
    
      model_bkl = Seq2Seq(encoder_bkl, decoder_bkl, device).to(device)
      model_bkl.apply(init_weights)
      optimizer = optim.Adam(model_bkl.parameters())
      criterion = nn.CrossEntropyLoss()
    
      model_bkl.train()
    
      # Initializing losses and accuracies
      training_epoch_loss = 0
      training_epoch_accuracy = 0
      validation_epoch_loss = 0
      validation_epoch_accuracy = 0
    
      num_epoch = 10
      CLIP = 1
    
      for epoch in range(num_epoch):
    
      #TRAINING BLOCK
        model_bkl.train()
        for _, (source, target) in enumerate(train_itersn):
            source, target = source.to(device), target.to(device)
            #print("target1: ", target)
            optimizer.zero_grad()
    
            output = model_bkl(source, target)
    
            output = output[1:].view(-1, output.shape[-1])
            target = target.permute(1,0)
            target = torch.reshape(target[1:], (-1,))
            #print("target2: ", target.shape)
            loss = criterion(output, target)
    
            # calculating accuracy
            predz = torch.argmax(output, dim=1)
            #print("pred: ", predz)
            #print("target3: ", target)
            non_pad_elements = (target != 0).nonzero(as_tuple=True)[0]
            #print("non_pad_elements: ", non_pad_elements)
            train_correct = predz[non_pad_elements] == target[non_pad_elements]
    
            #print("len(non_pad_elements): ", len(non_pad_elements))
            training_epoch_accuracy += train_correct.sum().item() / len(non_pad_elements)
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model_bkl.parameters(), CLIP)
    
            optimizer.step()
    
            training_epoch_loss += loss.item()
    
    
        training_epoch_loss = training_epoch_loss / len(train_itersn)
        training_epoch_accuracy = training_epoch_accuracy / len(train_itersn)
    
    
        #EVALUATION MODE ON
        model_bkl.eval()
    
        with torch.no_grad():
    
            for _, (source, target) in enumerate(valid_itersn):
                source, target = source.to(device), target.to(device)
    
                output = model_bkl(source, target, 0) # Turn off teacher forcing
    
                output = output[1:].view(-1, output.shape[-1])
                target = target.permute(1,0)
                target = torch.reshape(target[1:], (-1,))
    
                loss = criterion(output, target)
    
                validation_epoch_loss += loss.item()
                # Calculating accuracy
                predz = torch.argmax(output, dim=1)
                non_pad_elements = (target != 0).nonzero(as_tuple=True)[0]
                val_correct = predz[non_pad_elements] == target[non_pad_elements]
                validation_epoch_accuracy += val_correct.sum().item() / len(non_pad_elements)
    
        validation_epoch_accuracy = validation_epoch_accuracy / len(valid_itersn)
        validation_epoch_loss = validation_epoch_loss / len(valid_itersn)
    
        print(f'Epoch: {epoch+1:02} |Train Loss: {training_epoch_loss:.3f} | Train accuracy: {training_epoch_accuracy:.3f}|Val. Loss: {validation_epoch_loss:.3f} | Val accuracy: {validation_epoch_accuracy:.3f}')
        wandb.log({"train_loss":training_epoch_loss,"train_accuracy": training_epoch_accuracy,"val_loss":validation_epoch_loss,"val_accuracy":validation_epoch_accuracy},)
    
        # Emptying the cache after one complete run
        if epoch==num_epoch-1:
                torch.cuda.empty_cache()
    
    
    #RUNNING THE SWEEP
    wandb.agent(sweep_id, function=sweep_train, count=1)