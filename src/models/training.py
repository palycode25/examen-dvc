import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os

def main():
    input_folder = "data/processed_data"
    models_folder = "models"

    X_train = pd.read_csv(os.path.join(input_folder, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_folder, "y_train.csv")).values.ravel()

    with open(os.path.join(models_folder, "best_params.pkl"), "rb") as f:
        best_params = pickle.load(f)

    print(f"Entraînement avec les paramètres : {best_params}")

    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    with open(os.path.join(models_folder, "gbrt_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("Entraînement terminé : gbrt_model.pkl sauvegardé")

if __name__ == "__main__":
    main()
