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
    """Modified Bidirectional GRU Model to output the original hidden size."""
    def __init__(self, input_size=1280, hidden_size=1280, n_layers=1, batch_size=128):
        super(BiGRUModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # Bidirectional GRU but we will only use one direction's hidden state
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        # Linear layer to ensure output matches 1280 features if bidirectional effect needs reduction
        self.hidden_reduction = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, seq, candidates, return_hidden=False):
        # Run through the bidirectional GRU
        _, hidden = self.rnn(seq)
        
        # Use only one hidden state direction, or transform it to the original hidden size
        hidden = hidden[-2]  # Optionally, if both directions needed: hidden = self.hidden_reduction(torch.cat((hidden[-2], hidden[-1]), dim=1))
        
        # Compute similarity scores
        scores = torch.einsum("...bd,bcd->...bc", hidden, candidates)
        
        if return_hidden:
            return scores.squeeze(0), hidden.squeeze(0)
        return scores.squeeze(0)
    
class LSTMModel(nn.Module):
    """Modified LSTM Model to output the original hidden size."""
    def __init__(self, input_size=1280, hidden_size=1280, n_layers=1, batch_size=128):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        # Optional linear layer for dimension reduction if both directions are used
        self.hidden_reduction = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, seq, candidates, return_hidden=False):
        # Run through the LSTM
        _, (hidden, _) = self.rnn(seq)
        
        # Use only the forward hidden state, or reduce dimension if both directions are needed
        hidden = hidden[-2]  # For forward-only; use hidden = self.hidden_reduction(torch.cat((hidden[-2], hidden[-1]), dim=1)) for both directions
        
        # Compute similarity scores
        scores = torch.einsum("...bd,bcd->...bc", hidden, candidates)
        
        if return_hidden:
            return scores.squeeze(0), hidden.squeeze(0)
        return scores.squeeze(0)

class TransformerModel(nn.Module):
    """Modified Transformer-based Model to output the original hidden size."""
    def __init__(self, input_size=1280, hidden_size=1280, nhead=8, num_layers=2, candidate_len=3):
        super(TransformerModel, self).__init__()
        # Set batch_first=True for compatibility
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.hidden_reduction = nn.Linear(input_size, hidden_size)  # Ensures output matches the expected dimension

    def forward(self, seq, candidates, return_hidden=False):
        # Encode the sequence
        encoded_seq = self.transformer_encoder(seq)  # Shape: [batch_size, seq_len, hidden_size]
        last_hidden = encoded_seq[:, -1, :]  # Last hidden state, Shape: [batch_size, hidden_size]

        # Reduce to original hidden size if necessary
        last_hidden = self.hidden_reduction(last_hidden)  # Shape: [batch_size, input_size]

        # Compute similarity scores
        scores = torch.einsum("bd,bcd->bc", last_hidden, candidates)
        
        if return_hidden:
            return scores, last_hidden
        return scores