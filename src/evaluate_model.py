from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========\n")

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy*100:.2f}%")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix\n")
    print(cm)

    # Create images folder if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.colorbar()

    labels = ["Setosa", "Versicolor", "Virginica"]

    plt.xticks([0,1,2], labels, rotation=20)
    plt.yticks([0,1,2], labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j],
                     ha="center",
                     va="center",
                     color="black")

    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")
    plt.show()

    return predictions