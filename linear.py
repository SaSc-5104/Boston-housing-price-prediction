import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from evaluate import compute_metrics, plot_predictions

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_linear(model, X_test, y_test, feature_names, save_dir = "report"):
    y_pred = model.predict(X_test)
    print(f"\n LINEAR REGRESSION RESULTS:")
    metrics = compute_metrics(y_test, y_pred)
    for k,v in metrics.items():
        print(f"    {k}: {v:.4f}")
    plot_predictions(y_test, y_pred, title = "Linear Regression: Predicted vs Actual", save_path = f"{save_dir}/linear_predictions.png")
    return metrics, y_pred

def _plot_coefficients(coefs, feature_names, save_dir):
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_coefs = coefs[sorted_idx]
    sorted_names = []
    for i in sorted_idx:
        sorted_names.append(feature_names[i])
    colors = []
    for c in sorted_coefs:
        if c > 0:
            colors.append('steelblue')
        else:
            colors.append('tomato')
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_names, sorted_coefs, color=colors)
    plt.axhline(0, color= 'black', linewidth=0.8)
    plt.title("Linear Regression Feature Coefficients")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/linear_coefficients.png", dpi=150)
    plt.close()
    print(f"Coefficient plot saved to {save_dir}/linear_coefficients.png")


