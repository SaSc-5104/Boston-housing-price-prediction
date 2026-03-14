import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def plot_predictions(y_true, y_pred, title = "Predicted vs Actual", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.4, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Fit')
    plt.title(title)
    plt.xlabel("Actual MEDV ($1,000s)")
    plt.ylabel("Predicted MEDV ($1,000s)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    Saved: {save_path}")
    else:
        plt.show()

def plot_comparison(metrics_linear, metrics_neural, save_dir = "report"):
    metric_names = ["MSE", "RMSE", "MAE", "R2"]
    lin_vals = []
    for m in metric_names:
        lin_vals.append(metrics_linear[m])
    nn_vals = []
    for m in metric_names:
        nn_vals.append(metrics_neural[m])
    
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, lin_vals, width, label='Linear Regression', color='steelblue')
    bars2 = ax.bar(x + width/2, nn_vals, width, label='Neural Network', color='lightcoral')

    ax.set_title("Model Comparison Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha = "center", va = "bottom", fontsize = 8)

    plt.tight_layout()
    path = f"{save_dir}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def print_comparison_table(metrics_linear, metrics_neural):
    print("\n" + "=" * 50)
    print(f"{'Metric':<10} {'Linear Reression':>18} {'Neural Network':>15}")
    print("=" * 50)
    for m in metrics_linear.keys():
        label = "R²" if m == "R2" else m
        print(f"{label:<10} {metrics_linear[m]:>18.4f} {metrics_neural[m]:>15.4f}")
    print("=" * 50)