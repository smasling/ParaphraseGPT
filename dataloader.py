from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import json
import csv
import numpy as np

class ParaphrasesDataset(Dataset):
    def __init__(self):
        super().__init__()
        saved = np.load('paraphrases.npy')
        self.examples = []
        self.end_of_text_token = "<|endoftext|>"
        for p in saved:
          self.joke_list.append(p + self.end_of_text_token)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
