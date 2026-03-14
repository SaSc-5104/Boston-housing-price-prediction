import os
from data import load_data, preprocess
from linear import train_linear, evaluate_linear
from neural_model import train_neural, evaluate_neural
from evaluate import plot_comparison, print_comparison_table

SAVE_DIR = "report"
TEST_SIZE = 0.2
RANDOM_SEED = 42
NN_EPOCHS = 200
NN_LR = 1e-3
NN_BATCH = 32

def main():
    os.makedirs(SAVE_DIR, exist_ok = True)

    # Load and preprocess data
    print("\nLoading Data:")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df, test_size = TEST_SIZE, random_state = RANDOM_SEED)

    # Liner Regression
    print("\nLinear Regression:")
    lin_model = train_linear(X_train, y_train)
    metrics_linear, _ = evaluate_linear(lin_model, X_test, y_test, feature_names, save_dir = SAVE_DIR)

    # Neural Network
    print("\nNeural Network:")
    nn_model, history = train_neural (X_train, y_train, epochs = NN_EPOCHS, lr = NN_LR, batch_size = NN_BATCH, random_state = RANDOM_SEED)
    metrics_neural, _ = evaluate_neural(nn_model, X_test, y_test, history, save_dir = SAVE_DIR)

    # Comparison
    print("\nComparison:")
    print_comparison_table(metrics_linear, metrics_neural)
    plot_comparison(metrics_linear, metrics_neural, save_dir = SAVE_DIR)

    print(f"\nAll plots saved to ./{SAVE_DIR}/")
    print("DONE")

if __name__ == "__main__":
    main()
