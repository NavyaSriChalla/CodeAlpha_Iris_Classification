import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


def species_distribution():

    df = pd.read_csv("dataset/Iris.csv")

    plt.figure(figsize=(6,4))

    df["Species"].value_counts().plot(
        kind="bar",
        color=["#4CAF50", "#2196F3", "#FF9800"]
    )

    plt.title("Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("images/species_distribution.png")
    plt.show()


def correlation_heatmap():

    df = pd.read_csv("dataset/Iris.csv")

    df = df.drop("Id", axis=1)

    corr = df.iloc[:, :-1].corr()

    plt.figure(figsize=(6,5))

    plt.imshow(corr, cmap="Blues")

    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png")
    plt.show()


def feature_importance(model, X):

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    plt.figure(figsize=(6,4))

    importance.sort_values().plot(
        kind="barh",
        color="teal"
    )

    plt.title("Feature Importance")

    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.show()