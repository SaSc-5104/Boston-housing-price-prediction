import numpy as np 
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns = boston.feature_names)
    df["MEDV"] = boston.target
    return df

def preprocess(df, test_size = 0.2, random_state = 42):
    feature_names = []
    for col in df.columns:
        if col != "MEDV":
            feature_names.append(col)
    X = df[feature_names].values
    y = df["MEDV"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
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
    print(df.corr()["MEDV"].sort_values(ascending = False))
    