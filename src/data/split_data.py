import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    input_filepath = "data/raw_data/raw.csv"
    output_folder = "data/processed_data"

    df = pd.read_csv(input_filepath)

    # On retire la colonne date, non pertinente pour la modélisation
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # La cible est la dernière colonne : silica_concentrate
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(output_folder, exist_ok=True)

    X_train.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index=False)

    print(f"Split terminé : {X_train.shape[0]} lignes train / {X_test.shape[0]} lignes test")

if __name__ == "__main__":
    main()
