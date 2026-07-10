import joblib


def predict_flower(encoder):

    model = joblib.load("models/iris_model.pkl")

    print("\n========== FLOWER PREDICTION ==========\n")

    sl = float(input("Sepal Length : "))
    sw = float(input("Sepal Width  : "))
    pl = float(input("Petal Length : "))
    pw = float(input("Petal Width  : "))

    prediction = model.predict([[sl, sw, pl, pw]])

    flower = encoder.inverse_transform(prediction)

    print("\nPredicted Flower Species :", flower[0])