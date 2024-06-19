from torch import nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.RNN(300, 128, 1)

        self.main = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.main(x)
        return x