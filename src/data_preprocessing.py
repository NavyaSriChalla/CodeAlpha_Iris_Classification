import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess():

    # Load dataset
    df = pd.read_csv("dataset/Iris.csv")

    print("\n========== DATASET INFORMATION ==========\n")
    print(df.head())

    print("\nShape :", df.shape)

    print("\nColumns :")
    print(df.columns)

    print("\nMissing Values :")
    print(df.isnull().sum())

    print("\nStatistical Summary")
    print(df.describe())

    # Remove unnecessary column
    df.drop("Id", axis=1, inplace=True)

    # Encode target labels
    encoder = LabelEncoder()
    df["Species"] = encoder.fit_transform(df["Species"])

    X = df.drop("Species", axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, encoder, X
    