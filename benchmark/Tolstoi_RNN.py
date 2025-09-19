from pathlib import Path
from typing import List

import requests
import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark.utils import loss_win_condition, trial
from heavyball.utils import set_torch

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

app = typer.Typer()


class TolstoiDataset(Dataset):
    def __init__(self, text, seq_length=50, train=True, train_split=0.8):
        self.seq_length = seq_length

        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]

        # Split data
        split_idx = int(len(self.data) * train_split)
        if train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence
        x = torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long)
        # Target sequence (shifted by 1)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1], dtype=torch.long)
        return x, y


class CharRNN(nn.Module):
    def __init__(self, vocab_size=83, hidden_size=128, num_layers=2, seq_length=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Embedding (same as TF)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM layers (no dropout here! TF dropout was external via wrappers)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=0.0,  # disable internal LSTM dropout
            batch_first=True,
        )

        # Input/output dropout wrappers (20% like TF)
        self.input_dropout = nn.Dropout(0.2)  # input_keep_prob=0.8
        self.output_dropout = nn.Dropout(0.2)  # output_keep_prob=0.8

        # Output linear layer
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Hidden state
        self.hidden = None

    def forward(self, x):
        batch_size = x.size(0)

        # Init hidden if needed
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.hidden = self.init_hidden(batch_size)

        # Embedding + input dropout
        embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)

        # LSTM
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)

        # Detach hidden state (avoid backprop through history)
        self.hidden = tuple([h.detach() for h in self.hidden])

        # Apply output dropout
        lstm_out = self.output_dropout(lstm_out)

        # Reshape for linear layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        # Project to vocab
        output = self.fc(lstm_out)

        # Reshape back
        output = output.view(batch_size, self.seq_length, self.vocab_size)

        return output

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )

    def reset_hidden(self):
        self.hidden = None


def download_war_and_peace():
    """Download War and Peace text from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"

    response = requests.get(url)
    response.raise_for_status()
    return response.text


@app.command()
def main(
    method: List[str] = typer.Option(["qr"], help="Eigenvector method to use (for SOAP)"),
    dtype: List[str] = typer.Option(["float32"], help="Data type to use"),
    hidden_size: int = 128,
    batch: int = 50,
    steps: int = 0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    seq_length: int = 50,
    test_loader: bool = None,
):
    dtype = [getattr(torch, d) for d in dtype]

    # Download and prepare data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    text_file = data_dir / "war_and_peace.txt"
    if not text_file.exists():
        print("Downloading War and Peace...")
        text = download_war_and_peace()
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

    # Create datasets
    train_dataset = TolstoiDataset(text, seq_length=seq_length, train=True)
    test_dataset = TolstoiDataset(text, seq_length=seq_length, train=False)

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    model = CharRNN(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=2, seq_length=seq_length).cuda()
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    # Create data iterator
    data_iter = iter(train_loader)

    def data():
        nonlocal data_iter
        try:
            batch_data, batch_targets = next(data_iter)
        except StopIteration:
            # Reset iterator and hidden state when exhausted
            data_iter = iter(train_loader)
            model.reset_hidden()
            batch_data, batch_targets = next(data_iter)
        return batch_data.cuda(), batch_targets.cuda()

    # Custom loss function for sequence modeling
    def loss_fn(output, target):
        # output: (batch, seq, vocab)
        # target: (batch, seq)
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

        loss = F.cross_entropy(output, target, reduction="none")

        # Reshape to (batch, seq)
        loss = loss.view(target.size(0) // output.size(0), -1)

        # Average across timesteps (dim=1), keep batch dimension
        loss = loss.mean(dim=1)

        return loss.mean()

    trial(
        model,
        data,
        loss_fn,
        loss_win_condition(win_condition_multiplier * 0.0),
        steps,
        opt[0],
        weight_decay,
        failure_threshold=10,
        trials=trials,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    app()
