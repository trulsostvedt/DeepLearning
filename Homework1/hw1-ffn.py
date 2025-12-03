#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, t, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()
        
        n_classes = t
        
        # Define the layers
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
        
        #Dropout layer
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

            
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        in_dim = n_features
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size
        
        #Output layer
        self.output_layer = nn.Linear(in_dim, n_classes) 
        

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        h = x
        for layer in self.hidden_layers:
            h = layer(h)           
            h = self.activation(h) 
            h = self.dropout(h)     

        scores = self.output_layer(h)
        return scores
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(X)

    # Compute loss
    loss = criterion(logits, y)

    # Backward pass and optimization step
    loss.backward()

    # Update weights
    optimizer.step()

    # Return loss as a float
    return loss.item()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
    return preds

@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)

    preds = torch.argmax(logits, dim=1)
    correct = (preds == y).sum().item()
    total = y.size(0)
    acc = correct / total

    return loss.item(), acc


def run_experiment(
    n_classes, n_feats, dataset, train_X, train_y, dev_X, dev_y, test_X, test_y,
    batch_size, hidden_size, layers, learning_rate, l2_decay, dropout, activation, 
    optimizer_name, epochs,
):
    """
    Run a feedforward network experiment with the given hyperparameters.
    Returns a dictionary with training and validation losses and accuracies.
    """

    # Create DataLoader for batching
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    # Initialize model
    model = FeedforwardNetwork(n_classes, n_feats, hidden_size, layers, activation, dropout)

    # Define optimizer and loss function
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[optimizer_name]
    optimizer = optim_cls(
        model.parameters(), lr=learning_rate, weight_decay=l2_decay
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # Initial evaluation before training
    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)

    for epoch in range(1, epochs + 1):
        print(f"Training epoch {epoch}")
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print(
            f"train loss: {epoch_train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    _, test_acc = evaluate(model, test_X, test_y, criterion)

    # Find the best validation accuracy
    best_val_acc = max(valid_accs)

    results = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_accs": train_accs,
        "valid_accs": valid_accs,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }

    return results



def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz')
    parser.add_argument('--grid_search', action='store_true',
                        help="Run the 2.2(a) grid search over widths/lr/dropout/l2.")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    # Load dataset
    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # =========================
    # 2.2(a): GRID SEARCH-MODUS
    # =========================
    if opt.grid_search:
    # ====== 2.2(a): GRID SEARCH ======
        widths = [16, 32, 64, 128, 256]
        learning_rates = [0.0005, 0.001, 0.002, 0.005]
        dropouts = [0.0, 0.3]
        l2_values = [0.0, 1e-4]

        optimizer_name = opt.optimizer
        activation = opt.activation

        all_results = []
        global_best = None

        for width in widths:
            for lr in learning_rates:
                for dropout in dropouts:
                    for l2 in l2_values:
                        print("=" * 80)
                        print(f"Width={width} | lr={lr} | dropout={dropout} | l2={l2}")
                        start = time.time()

                        results = run_experiment(
                            n_classes=n_classes,
                            n_feats=n_feats,
                            dataset=dataset,
                            train_X=train_X,
                            train_y=train_y,
                            dev_X=dev_X,
                            dev_y=dev_y,
                            test_X=test_X,
                            test_y=test_y,
                            batch_size=64,
                            hidden_size=width,
                            layers=1,
                            learning_rate=lr,
                            l2_decay=l2,
                            dropout=dropout,
                            activation=activation,
                            optimizer_name=optimizer_name,
                            epochs=opt.epochs,
                        )

                        elapsed = time.time() - start

                        config_result = {
                            "width": width,
                            "learning_rate": lr,
                            "dropout": dropout,
                            "l2_decay": l2,
                            "best_val_acc": results["best_val_acc"],
                            "test_acc": results["test_acc"],
                            "time_sec": elapsed
                        }
                        all_results.append(config_result)

                        if (global_best is None or
                            config_result["best_val_acc"] > global_best["best_val_acc"]):
                            global_best = config_result

        # ==========================
        # WRITE RESULTS TO CSV FILE
        # ==========================
        import csv
        csv_filename = f"gridsearch_results_{optimizer_name}_{activation}.csv"

        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "width", "learning_rate", "dropout", "l2_decay",
                    "best_val_acc", "test_acc", "time_sec"
                ]
            )
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nGrid search completed. Results saved to {csv_filename}")
        print("Best config overall:")
        print(global_best)
        return


    # ==============================
    # Vanlig enkelt-run (ikke grid)
    # ==============================
    start = time.time()

    # Run experiment with given hyperparameters
    results = run_experiment(
        n_classes=n_classes,
        n_feats=n_feats,
        dataset=dataset,
        train_X=train_X,
        train_y=train_y,
        dev_X=dev_X,
        dev_y=dev_y,
        test_X=test_X,
        test_y=test_y,
        batch_size=opt.batch_size,
        hidden_size=opt.hidden_size,
        layers=opt.layers,
        learning_rate=opt.learning_rate,
        l2_decay=opt.l2_decay,
        dropout=opt.dropout,
        activation=opt.activation,
        optimizer_name=opt.optimizer,
        epochs=opt.epochs,
    )

    train_losses = results["train_losses"]
    valid_losses = results["valid_losses"]
    train_accs = results["train_accs"]
    valid_accs = results["valid_accs"]
    best_val_acc = results["best_val_acc"]
    test_acc = results["test_acc"]

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print(f"Final test acc: {test_acc:.4f}")

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    model_name = "ffn"
    plot_epochs = list(range(len(train_losses)))

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }
    plot(plot_epochs, losses, filename=f'{model_name}-training-loss-{config}.pdf')
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    val_accuracy = {"Valid Accuracy": valid_accs}
    plot(plot_epochs, val_accuracy, filename=f'{model_name}-validation-accuracy-{config}.pdf')


if __name__ == '__main__':
    main()
