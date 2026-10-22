"""Train, compare and save the best admission-chance model."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("D:\Agentic AI Learning\Machine Learning\Regression\Linear Regression\Data\Admission_Predict.csv")
df.columns = df.columns.str.strip()          # 'LOR ' / 'Chance of Admit ' have trailing spaces
df = df.drop(columns=["Serial No."])         # row ID, not a real feature

X = df.drop(columns="Chance of Admit")
y = df["Chance of Admit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

candidates = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}

results = {}
for name, m in candidates.items():
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring="r2").mean()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results[name] = {
        "cv_r2": round(cv, 4),
        "test_r2": round(r2_score(y_test, pred), 4),
        "test_mae": round(mean_absolute_error(y_test, pred), 4),
    }
    print(name, results[name])

best_name = max(results, key=lambda k: results[k]["cv_r2"])
best = candidates[best_name]
best.fit(X, y)  # refit on all data for the deployed model

joblib.dump({"model": best, "features": list(X.columns)}, "model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"best": best_name, "results": results}, f, indent=2)
print("Saved model:", best_name)
