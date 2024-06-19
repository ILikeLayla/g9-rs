import torch
from model import *
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer

product_name = "dyes"
original_price = 45.0

GLOVE = GloVe(name='840B', dim=300)

net = RNN()
net.load_state_dict(torch.load('model.pth'))

vector = GLOVE.get_vecs_by_tokens(product_name).reshape(1,-1)

percentage = net(vector).item()

print(f"rate of increase: {percentage * 100}%")
print(f"new price: {original_price * (1 + percentage)}")