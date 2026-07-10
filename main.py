from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualization import species_distribution, correlation_heatmap, feature_importance
from src.predictor import predict_flower

print("=" * 50)
print("      IRIS FLOWER CLASSIFICATION")
print("=" * 50)

# Load Data
X_train, X_test, y_train, y_test, encoder, X = load_and_preprocess()

# Visualizations
species_distribution()
correlation_heatmap()

# Train Model
model = train_model(X_train, y_train)

# Evaluate Model
evaluate_model(model, X_test, y_test)

# Feature Importance
feature_importance(model, X)

# Predict New Flower
predict_flower(encoder)

print("\nProject Completed Successfully!")