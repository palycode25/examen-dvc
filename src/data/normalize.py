import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

def main():
    input_folder = "data/processed_data"
    models_folder = "models"

    X_train = pd.read_csv(os.path.join(input_folder, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_folder, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled_df.to_csv(os.path.join(input_folder, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(input_folder, "X_test_scaled.csv"), index=False)

    os.makedirs(models_folder, exist_ok=True)
    with open(os.path.join(models_folder, "scaling_model.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Normalisation terminée : X_train_scaled.csv et X_test_scaled.csv créés")

if __name__ == "__main__":
    main()
