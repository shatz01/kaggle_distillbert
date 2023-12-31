import torch
import numpy as np

class DAIGTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels = None):
        self.tokenized_data = tokenized_data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros(self.tokenized_data.input_ids.shape[0], dtype="int")
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return self.labels.shape[0]
    
    @classmethod
    def create_tokenized_dataset(cls, tknzr, df):
        tokenized_data = tknzr(
            df.text.tolist(), 
            max_length=tknzr.model_max_length, 
            padding="max_length", 
            return_tensors="pt",
            truncation=True
        )
        if "label" in df:
            labels = df.label.values
        else:
            labels = None
        return cls(tokenized_data, labels=labels)
