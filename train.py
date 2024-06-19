from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer
from torch import nn
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import *

GLOVE = GloVe(name='840B', dim=300)
BATCH_SIZE = 32
# ACCEPTABLE_RANGE = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DeltaPriceDataset(Dataset):
    def __init__(self, if_train=True) -> None:
        super().__init__()
        # self.tokenizer = get_tokenizer('basic_english')
        df = pd.read_csv(r"C:\code\g9-rs\data\data.csv")
        if if_train:
            df = df[:-60]
        else:
            df = df[-60:]
        ori_price = np.array(df["ori_price"])
        recycle_price = np.array(df["recycle_price"])
        self.delta_price = ((recycle_price - ori_price)/ori_price).tolist()
        self.product_name = df["eng_name"].tolist()
    
    def __len__(self):
        return len(self.product_name)
    
    def __getitem__(self, idx):
        product_name = self.product_name[idx]
        vector = GLOVE.get_vecs_by_tokens(product_name)
        delta_price = self.delta_price[idx]
        return vector, delta_price

train_dataset = DeltaPriceDataset(if_train=True)
test_dataset = DeltaPriceDataset(if_train=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
net = RNN().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

losses = []

for epoch in range(5000):
    net.train()
    for x, y in train_dataloader:
        x = x.to(device).float()
        y = y.to(device).float()
        optimizer.zero_grad()

        y_pred = net(x).squeeze(1)
        # print(y_pred.shape)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    if epoch % 100 == 0:
        # correct = 0
        # total = 0
        # net.eval()
        # with torch.no_grad():
        #     for x, y in test_dataloader:
        #         x = x.to(device).float()
        #         y = y.to(device).float()
        #         y_pred = net(x).squeeze(1)
        #         error = (y_pred - y) / y
        #         print(error)
        #         correct += torch.sum(torch.abs(error) < ACCEPTABLE_RANGE).item()
        #         total += y.shape[0]
        # print(correct)
        # print(total)
        # print(f"Epoch {epoch}, Loss: {loss.item()}, Acc: {correct/total}")
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(net.state_dict(), "./model.pth")


plt.plot(losses, label="Train Loss")
plt.legend(loc="upper right")
plt.show()
