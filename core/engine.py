"""OncoScan engine: Wisconsin breast-cancer data, model arena with cross-validation, explanations, similar cases, records."""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DB = Path(__file__).resolve().parent.parent / "users.db"
MEAN_FEATS = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"]


def load():
    d = load_breast_cancer(as_frame=True)
    X = d.data
    y = (d.target == 0).astype(int)  # 1 = malignant
    return X, y


MODELS = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=3, random_state=0), ensemble=False)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=0),
    "k-NN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "Neural Net (MLP)": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=0)),
}


def arena(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for name, mk in MODELS.items():
        t = time.time()
        p = cross_val_predict(mk(), X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        out[name] = {"proba": p, "auc": roc_auc_score(y, p), "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "precision": tp / (tp + fp), "roc": roc_curve(y, p), "time": time.time() - t}
    return out


def fit_final(X, y, name):
    return MODELS[name]().fit(X, y)


def explain_linear(X, y, row):
    """Contribution of each feature to the log-odds for a standardized logistic regression (transparent model)."""
    m = MODELS["Logistic Regression"]().fit(X, y)
    sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    z = sc.transform(row[X.columns])[0]
    contrib = pd.Series(z * lr.coef_[0], index=X.columns)
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index), lr.intercept_[0]


def similar_cases(X, y, row, k=5):
    sc = StandardScaler().fit(X)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(X))
    d, i = nn.kneighbors(sc.transform(row[X.columns]))
    out = X.iloc[i[0]][MEAN_FEATS[:4]].copy()
    out.insert(0, "diagnosis", np.where(y.iloc[i[0]] == 1, "Malignant", "Benign"))
    out.insert(1, "similarity", np.round(1 / (1 + d[0]), 3))
    return out


def pca(X, n=3):
    return PCA(n, random_state=0).fit_transform(StandardScaler().fit_transform(X))


# ---------------- patient records (per user) ----------------
def _con():
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, patient TEXT, age INTEGER, prob REAL, label TEXT, model TEXT, features TEXT, notes TEXT, ts REAL)")
    return con


def save_record(uid, patient, age, prob, label, model, features: dict, notes=""):
    with _con() as c:
        c.execute("INSERT INTO records (user_id, patient, age, prob, label, model, features, notes, ts) VALUES (?,?,?,?,?,?,?,?,?)", (uid, patient, age, prob, label, model, json.dumps(features), notes, time.time()))


def list_records(uid):
    with _con() as c:
        return pd.read_sql_query("SELECT id, patient, age, prob, label, model, notes, ts FROM records WHERE user_id=? ORDER BY ts DESC", c, params=(uid,))


def delete_record(uid, rid):
    with _con() as c:
        c.execute("DELETE FROM records WHERE id=? AND user_id=?", (rid, uid))


def risk_band(p):
    if p >= 0.85: return "High risk — malignant pattern", "red"
    if p >= 0.5: return "Elevated risk — likely malignant", "amber"
    if p >= 0.2: return "Indeterminate — recommend follow-up", "amber"
    return "Low risk — benign pattern", "green"
