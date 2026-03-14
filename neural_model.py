import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from evaluate import compute_metrics, plot_predictions

class MLP(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_neural(X_train, y_train, epochs=200, lr=1e-3, batch_size=32, val_split=0.1, random_state=42):
    torch.manual_seed(random_state)
    n_val = int(len(X_train) * val_split)
    idx = np.random.RandomState(random_state).permutation(len(X_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = MLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        train_loss = np.mean(batch_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    print("Neural network training complete.")
    return model, history


def evaluate_neural(model, X_test, y_test, history, save_dir="report"):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_tensor).numpy()

    print("\nNEURAL NETWORK RESULTS:")
    metrics = compute_metrics(y_test, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_predictions(
        y_test, y_pred,
        title="Neural Network: Predicted vs Actual",
        save_path=f"{save_dir}/neural_predictions.png"
    )

    _plot_loss_curve(history, save_dir)

    return metrics, y_pred


def _plot_loss_curve(history, save_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss", color="steelblue")
    plt.plot(history["val_loss"],  label="Val Loss",   color="tomato", linestyle="--")
    plt.title("Neural Network — Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/neural_loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/neural_loss_curve.png")
