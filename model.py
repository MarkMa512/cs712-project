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


# if __name__ == "__main__":
#     model = SimpleModel()
#     breakpoint()
#     # print(model)

class BiGRUModel(nn.Module):
    """Bidirectional GRU Model"""
    def __init__(self, input_size=1280, hidden_size=1280, n_layers=1, batch_size=128):
        super(BiGRUModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # Bidirectional GRU doubles the hidden size
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        # Linear layer to match candidates' dimension with bidirectional hidden size
        self.candidate_transform = nn.Linear(input_size, hidden_size * 2)

    def forward(self, seq, candidates, return_hidden=False):
        # Run through the GRU
        _, hidden = self.rnn(seq)
        
        # Concatenate hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1).unsqueeze(0)  # Shape: [1, batch_size, hidden_size * 2]
        
        # Adjust candidates to match hidden dimension (hidden_size * 2)
        candidates = self.candidate_transform(candidates)  # Shape: [batch_size, candidate_len, hidden_size * 2]
        
        # Compute similarity scores
        scores = torch.einsum("...bd,bcd->...bc", hidden, candidates)
        
        if return_hidden:
            return scores.squeeze(0), hidden.squeeze(0)
        return scores.squeeze(0)

class LSTMModel(nn.Module):
    """LSTM Model"""
    def __init__(self, input_size=1280, hidden_size=1280, n_layers=1, batch_size=128):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False
        )

    def forward(self, seq, candidates, return_hidden=False):
        _, (hidden, _) = self.rnn(seq)
        scores = torch.einsum("...bd,bcd->...bc", hidden, candidates)
        
        if return_hidden:
            return scores.squeeze(0), hidden.squeeze(0)
        return scores.squeeze(0)

class TransformerModel(nn.Module):
    """Transformer-based Model"""
    def __init__(self, input_size=1280, hidden_size=1280, nhead=8, num_layers=2, candidate_len=3):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_size, candidate_len)

    def forward(self, seq, candidates, return_hidden=False):
        # Encode sequence
        encoded_seq = self.transformer_encoder(seq)
        last_hidden = encoded_seq[:, -1, :].unsqueeze(1)
        scores = torch.einsum("...bd,bcd->...bc", last_hidden, candidates)

        if return_hidden:
            return scores.squeeze(0), last_hidden.squeeze(0)
        return scores.squeeze(0)