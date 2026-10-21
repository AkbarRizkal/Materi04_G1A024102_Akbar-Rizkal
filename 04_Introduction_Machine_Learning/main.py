from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA = Path(__file__).parent / "data" / "student_progress.csv"

def main():
    df = pd.read_csv(DATA)
    features = ["kehadiran_pct", "nilai_tugas"]
    X, y = df[features], df["status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("Akurasi pada data uji:", round(accuracy_score(y_test, prediction), 3))
    print(classification_report(y_test, prediction, zero_division=0))

    example = pd.DataFrame([{"kehadiran_pct": 72, "nilai_tugas": 58}])
    print("Prediksi contoh:", model.predict(example)[0])
    print("Jenis pembelajaran: supervised learning karena data memiliki label.")
    print("Catatan: dataset kecil dan sintetis; hasil tidak layak dipakai untuk keputusan nyata.")

if __name__ == "__main__":
    main()
