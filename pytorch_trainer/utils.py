import torch
import torch.nn as nn


def squareplus(x, b=1.52382103):
    return torch.mul(0.5, torch.add(x, torch.sqrt(torch.add(torch.square(x), b))))


class SquarePlus(nn.Module):
    def __init__(self, b=1.52382103):
        super().__init__()
        self.b = b

    def forward(self, x):
        return squareplus(x, self.b)
    

# A simple class for early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, filename='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_uncertainty = False
        self.val_loss_min = torch.inf
        self.delta = delta
        self.filename = filename

    def __call__(self, val_loss, model):
        if val_loss != val_loss:
            #nan check
            if self.verbose:
                print(f'Validation loss nan')
            self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.train_uncertainty = True
            if self.counter >= (self.patience+2):
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.filename)
        self.val_loss_min = val_loss

