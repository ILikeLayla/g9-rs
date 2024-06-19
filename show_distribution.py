from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchtext.vocab import GloVe
import pandas as pd

GLOVE = GloVe(name='840B', dim=300)

df = pd.read_csv(r"C:\code\g9-rs\data\data.csv")
words = df["eng_name"].tolist() + ["recycle"]

tsne = TSNE(n_components=2, init="random", learning_rate=200)
vectors = [GLOVE.get_vecs_by_tokens(word).tolist() for word in words]

# print(vectors[0].shape)
Y = tsne.fit_transform(vectors)
Y, recycle = Y[:-1], Y[-1]

plt.figure(figsize=(12, 8))
plt.axis('off')
plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.text(x, y, label, fontsize=5)

plt.scatter(recycle[0], recycle[1], c="red", marker="x")
plt.text(recycle[0], recycle[1], "recycle", fontsize=12, c="red")

plt.show()