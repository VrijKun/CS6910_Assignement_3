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


# Vanilla implimenststsiomn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1 , cell_type="lstm", p = 0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type=cell_type
        self.dropout=nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)

        if cell_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,dropout=p)
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,dropout=p)
        elif cell_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,dropout=p)
        else:
            raise ValueError("Invalid cell type selected: {}".format(cell_type))

    def forward(self, x):

        x=x.permute(1,0)
        embedding=self.dropout(self.embedding(x))
        if self.cell_type=="lstm":
         outputs,(hidden,cell)=self.rnn(embedding)
        else:
           hidden, cell = self.rnn(embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1, cell_type="lstm",p = 0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout=nn.Dropout(p)
        self.embedding=nn.Embedding(input_size, embedding_size)
        self.cell_type=cell_type

        self.output_size = output_size
        if cell_type == "gru":
            self.rnn = nn.GRU(output_size, hidden_size, num_layers)
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,dropout=p)
        elif cell_type == "rnn":
            self.rnn = nn.RNN(output_size, hidden_size, num_layers)
        else:
            raise ValueError("Invalid cell type selected: {}".format(cell_type))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden,cell):
       # shape of x: (N) but we want (1,N)
       x=x.unsqueeze(0)

       embedding= self.dropout(self.embedding(x))
       #embedding shape: (1,N,hidden_size)

       if self.cell_type=="lstm":
        outputs,(hidden,cell) = self.rnn(embedding, (hidden,cell))
       else:
         outputs, hidden = self.rnn(embedding, hidden)
       #shape of outputs: (1,N,hidden_size)


       predictions=self.fc(outputs)
       #shape of predictions: (1,N,length_of_vocab)

       predictions = predictions.squeeze(0)

       if self.cell_type=="lstm":
         return predictions, hidden, cell
       else:
         return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self,encoder,decoder,cell_type="lstm"):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]

        target_len = target.shape[1]
        target_vocab_size=self.decoder.output_size

        outputs=torch.zeros(target_len,batch_size,target_vocab_size).to(device)

        cell,hidden=self.encoder(source)

        target = target.permute(1,0)
        x = target[0,:]
        for t in range(1,target_len):
          if self.cell_type=="lstm":
           output,hidden,cell =self.decoder(x,hidden,cell)
          else:
            output, hidden = self.decoder(x,hidden,cell)

          outputs[t] = output

          best_guess = output.argmax(1)

          x= target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs








if __name__ == '__main__':    
  
    parser = argparse.ArgumentParser(description='Impliment Seqence to sequence model RNN, LSTM, GRU andapplu Attention in it')
    parser.add_argument('--wandb_entity', type=str, default='ed23d015', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='DL_Assignment_3', help='Name of the wandb project')
    parser.add_argument('--lr', type=int, default=0.001, help='Learning rate suggestion out of [0.001,0.0001]')
    parser.add_argument('--layer_size', type=int, default=3, help='No. of Hidden layer for the run out of [1,2,3,4]')
    parser.add_argument('--cell_type', type=str, default='gru', help='Cell type out of ["rnn","lstm","gru"]')
    parser.add_argument('--hidden_layers', type=int, default=256, help='Size of Hidden layer for the run [64,128,256]')
    parser.add_argument('--dropout', type=int, default=0.4, help='choices:[0,0.2,0.4,0.6]')
    

    args = parser.parse_args()


    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'values': [args.lr]
            },
            'layer_size': {
                'values': [args.layer_size]
            },
            'cell_type':{
                'values':[args.cell_type]
            },
            'dropout':{
                'values':[args.dropout]
            },
            'hidden_layers':{
                'values':[args.hidden_layers]
            },

        }
    }

    sweep_id = wandb.sweep(sweep_config, entity='ed23d015', project="DL_Assignment_3")

    def sweep_train():
    # Default values for hyper-parameters we're going to sweep over
        config_defaults = {
            'lr':0.001,
            'layer_size':3,
            'cell_type':'gru',
            'dropout':0.4,
            'hidden_layers':256,
        }

        # Initialize a new wandb run
        wandb.init(project='DL_Assignment_3', entity='ed23d015',config=config_defaults)
        wandb.run.name = 'cell:'+ str(wandb.config.cell_type)+' ;lr:'+str(wandb.config.lr)+ ' ;layer_size:'+str(wandb.config.layer_size)+ ' ;dropout:'+str(wandb.config.dropout)+' ;hidden:'+str(wandb.config.hidden_layers)

        config = wandb.config
        lr = config.lr
        layer_size = config.layer_size
        cell_type = config.cell_type
        hidden_layers = config.hidden_layers
        dropout = config.dropout

        encoder_embedding_size  = 512
        decoder_embedding_size  = 512

        # model_bkl training here

        input_size_encoder = len(eng_token_map)
        output_size=input_size_decoder=len(hin_token_map)

        encoder_embedding_size = 29
        decoder_embedding_size = 68
        encoder_net_bkl = Encoder(input_size_encoder, encoder_embedding_size, hidden_layers, layer_size, cell_type, dropout).to(device)
        decoder_net_bkl = Decoder(input_size_decoder, decoder_embedding_size, hidden_layers, output_size, layer_size, cell_type, p = dropout).to(device)
        model_bkl = Seq2Seq(encoder_net_bkl,decoder_net_bkl ,cell_type).to(device)
        criterion = nn.CrossEntropyLoss()
        import math
        import time

        optimizer = optim.Adam(model_bkl.parameters(),lr=lr)
        model_bkl.train()

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

                optimizer.zero_grad()

                output = model_bkl(source, target)
                output = output[1:].view(-1, output.shape[-1])
                target = target.permute(1,0)
                target = torch.reshape(target[1:], (-1,))
                loss = criterion(output, target)

                # calculate accuracy
                preds = torch.argmax(output, dim=1)
                non_pad_elements = (target != 0).nonzero(as_tuple=True)[0]
                train_correct = preds[non_pad_elements] == target[non_pad_elements]
                training_epoch_accuracy += train_correct.sum().item() / len(non_pad_elements)
                loss.backward()

                #READ IF WE CAN PASS CLIP AS A HYPERPARAMETER
                torch.nn.utils.clip_grad_norm_(model_bkl.parameters(), CLIP)

                optimizer.step()

                training_epoch_loss += loss.item()


            training_epoch_loss = training_epoch_loss / len(train_itersn)
            training_epoch_accuracy = training_epoch_accuracy / len(train_itersn)


            #EVALUATION MODE
            model_bkl.eval()

            with torch.no_grad():

                for _, (source, target) in enumerate(valid_itersn):
                    source, target = source.to(device), target.to(device)

                    output = model_bkl(source, target, 0) #turn off teacher forcing

                    output = output[1:].view(-1, output.shape[-1])
                    target = target.permute(1,0)
                    target = torch.reshape(target[1:], (-1,))

                    loss = criterion(output, target)

                    validation_epoch_loss += loss.item()
                    # calculate accuracy
                    preds = torch.argmax(output, dim=1)
                    non_pad_elements = (target != 0).nonzero(as_tuple=True)[0]
                    val_correct = preds[non_pad_elements] == target[non_pad_elements]
                    validation_epoch_accuracy += val_correct.sum().item() / len(non_pad_elements)

            validation_epoch_loss = validation_epoch_loss / len(valid_itersn)
            validation_epoch_accuracy = validation_epoch_accuracy / len(valid_itersn)

            print(f'Epoch: {epoch+1:02} |Train Loss: {training_epoch_loss:.3f} | Train accuracy: {training_epoch_accuracy:.3f}|Val. Loss: {validation_epoch_loss:.3f} | Val accuracy: {validation_epoch_accuracy:.3f}')
            wandb.log({"train_loss":training_epoch_loss,"train_accuracy": training_epoch_accuracy,"val_loss":validation_epoch_loss,"val_accuracy":validation_epoch_accuracy},)
            #emptying the cache after one complete run
            if epoch==num_epoch-1:
                    torch.cuda.empty_cache()


    #RUNNING THE SWEEP
    wandb.agent(sweep_id, function=sweep_train, count=40)
