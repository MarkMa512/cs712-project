import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=1280, n_layers=1, batch_size=128):
        super(SimpleModel, self).__init__()
        self.batch_size = batch_size
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False
        )

    def forward(self, seq, candidates, return_hidden=False):
        _, hidden = self.rnn(seq)
        # hidden: 1, batch_size, hidden_size
        # candidates: batch_size, candidate_len, hidden_size
        scores = torch.einsum("...bd,bcd->...bc", hidden, candidates)

        if return_hidden:
            return scores.squeeze(0), hidden.squeeze(0)

        return scores.squeeze(0)


if __name__ == "__main__":
    model = SimpleModel()
    breakpoint()
    # print(model)
