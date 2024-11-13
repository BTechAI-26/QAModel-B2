# Combined Script for Training and Evaluation

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

# Step 2: Load Data
formatted_output = pd.read_csv("/mnt/data/formatted_output.csv")
formatted_output[['Question', 'Answer']] = formatted_output['Q&A'].str.split(': ', expand=True, n=1)

# Step 3: Define Dataset Class
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = str(self.data['Question'][index])
        answer = str(self.data['Answer'][index])
        inputs = self.tokenizer(question, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(answer, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")['input_ids']
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

# Step 4: Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # adjust num_labels based on task

# Step 5: Prepare Data Loaders
train_data, val_data = train_test_split(formatted_output, test_size=0.1)
train_dataset = QADataset(train_data, tokenizer, max_len=128)
val_dataset = QADataset(val_data, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Step 6: Define Training Loop
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model = model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(data_loader)

# Step 7: Training and Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 3
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = eval_model(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Step 8: Save Model
model.save_pretrained('./model_checkpoint')
tokenizer.save_pretrained('./model_checkpoint')
