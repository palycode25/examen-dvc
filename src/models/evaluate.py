import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

def main():
    processed_folder = "data/processed_data"
    models_folder = "models"
    metrics_folder = "metrics"
    data_folder = "data"

    X_test = pd.read_csv(os.path.join(processed_folder, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(processed_folder, "y_test.csv")).values.ravel()

    with open(os.path.join(models_folder, "gbrt_model.pkl"), "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {"mse": mse, "r2": r2}
    print(f"Scores : {scores}")

    os.makedirs(metrics_folder, exist_ok=True)
    with open(os.path.join(metrics_folder, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    prediction_df = pd.DataFrame({"y_true": y_test, "y_pred": predictions})
    prediction_df.to_csv(os.path.join(data_folder, "prediction.csv"), index=False)

    print("Évaluation terminée : scores.json et prediction.csv sauvegardés")

if __name__ == "__main__":
    main()
