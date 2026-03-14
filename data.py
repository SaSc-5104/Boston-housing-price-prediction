import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    df["MEDV"] = target
    return df


def preprocess(df, test_size=0.2, random_state=42):
    feature_names = [col for col in df.columns if col != "MEDV"]
    X = df[feature_names].values
    y = df["MEDV"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset loaded: {len(df)} samples, {len(feature_names)} features")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    df = load_data()
    print(df.describe())
    print("\nFeature correlations with MEDV:")
    print(df.corr()["MEDV"].sort_values(ascending=False))
