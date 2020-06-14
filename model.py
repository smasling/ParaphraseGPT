import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from dataloader import ParaphrasesDataset
from torch.utils.data import DataLoader
import numpy as np
from utils import *

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
EOT_TOKEN = "<|endoftext|>"


def train(tokenizer, m, device, loader):
    model = m.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0

    tmp_phrases = None
    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(EPOCHS):

        print("EPOCH" + epoch)

        for idx,phrase in enumerate(loader):

            #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            p = torch.tensor(tokenizer.encode(phrase[0])).unsqueeze(0).to(device)
            #Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if p.size()[1] > MAX_SEQ_LEN:
                continue

            #The first joke sequence in the sequence
            if not torch.is_tensor(tmp_phrases):
                tmp_phrases = p
                continue
            else:
                #The next joke does not fit in so we process the sequence and leave the last joke
                #as the start for next sequence
                if tmp_phrases.size()[1] + p.size()[1] > MAX_SEQ_LEN:
                    final = tmp_phrases
                    tmp_phrases = p
                else:
                    #Add the joke to sequence, continue and try to add more
                    tmp_phrases = torch.cat([tmp_phrases, p[:,1:]], dim=1)
                    print(p[:,1:]) #TODO:comment out later
                    continue
            ################## Sequence ready, process it trough the model ##################
            aux_loss = task_loss(tokenizer.decode(final).split(EOT_TOKEN))
            outputs = model(final, labels=final)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 10:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(models_folder, f"model_{epoch}.pt"))

def generate(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    MODEL_EPOCH = 0
    models_folder = "trained_models"
    model_path = os.path.join(models_folder, f"model_{MODEL_EPOCH}.pt")
    model.load_state_dict(torch.load(model_path))

    model.eval()
    # if os.path.exists(jokes_output_file_path):
    #     os.remove(jokes_output_file_path)
    para_num = 0
    with torch.no_grad():
    # for joke_idx in range(1000):
        joke_finished = False
        phrase = "SOMETHING EHRE[SEP]"
        cur_ids = torch.tensor(tokenizer.encode(phrase)).unsqueeze(0).to(device)
        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                joke_finished = True
                break



        if joke_finished:

            # joke_num = joke_num + 1

            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)
            print(output_text)

            # with open(jokes_output_file_path, 'a') as f:
            #     f.write(f"{output_text} \n\n")



def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = model.to(device)
    dataset = ParaphrasesDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    train(tokenizer, model, device, loader)


if __name__ == "__main__":
    main()