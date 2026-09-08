import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

def main():
    input_folder = "data/processed_data"
    models_folder = "models"

    X_train = pd.read_csv(os.path.join(input_folder, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_folder, "y_train.csv")).values.ravel()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }

    model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres trouvés : {best_params}")

    os.makedirs(models_folder, exist_ok=True)
    with open(os.path.join(models_folder, "best_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)

    print("GridSearch terminé : best_params.pkl sauvegardé")

if __name__ == "__main__":
    main()
