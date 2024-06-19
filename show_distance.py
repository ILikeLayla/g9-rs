import matplotlib.pyplot as plt
from torchtext.vocab import GloVe
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

GLOVE = GloVe(name='840B', dim=300)

df = pd.read_csv(r"C:\code\g9-rs\data\data.csv")
words = df["eng_name"].tolist() + ["recycle"]

tsne = TSNE(n_components=1, init="random", learning_rate=200)
vectors = [GLOVE.get_vecs_by_tokens(word).numpy() for word in words]
vectors, recycle_vector = vectors[:-1], vectors[-1]

Y = np.linalg.norm(vectors - recycle_vector, axis=1)
X = tsne.fit_transform(vectors)

plt.scatter(X, Y)
plt.show()