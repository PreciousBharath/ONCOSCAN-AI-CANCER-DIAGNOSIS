"""OncoScan AI — explainable breast-cancer diagnosis support (Streamlit, scikit-learn)."""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import auth, ui
from core.engine import MEAN_FEATS, MODELS, arena, delete_record, explain_linear, fit_final, list_records, load, pca, risk_band, save_record, similar_cases

st.set_page_config(page_title="OncoScan AI · Breast Cancer Diagnosis", page_icon="🎗️", layout="wide")
ui.inject_css()
APP, ICON = "OncoScan AI", "🎗️"
if not auth.auth_page(APP, "Explainable machine learning that supports breast-cancer diagnosis from cell-nucleus measurements.",
                      ["🧪 6 models compared with 5-fold cross-validation", "🔍 Transparent feature-level explanations",
                       "👥 Similar-case retrieval from 569 biopsies", "📋 Secure per-user patient records & reports"], ICON,
                      demo={"name": "Dr. Demo", "email": "demo@oncoscan.ai", "password": "Demo@1234"}):
    st.stop()


@st.cache_data
def data():
    return load()


@st.cache_resource(show_spinner="Cross-validating 6 models…")
def models():
    X, y = data()
    res = arena(X, y)
    return res, {n: fit_final(X, y, n) for n in MODELS}


X, y = data()
res, fitted = models()
best = max(res, key=lambda n: res[n]["auc"])
uid = st.session_state.user["id"]
page = ui.sidebar(APP, ICON, ["🩺 Diagnose", "📋 Patient Records", "🧪 Model Arena", "🔬 Data Explorer", "📁 Batch Screening", "👤 Account"])
st.sidebar.caption("⚕️ Decision-support demo for education — not a medical device.")

if page == "🩺 Diagnose":
    ui.hero("Diagnostic Assistant", "Enter fine-needle-aspirate measurements and get an explainable malignancy risk", "🩺")
    ss = st.session_state
    if "sample" not in ss:
        ss.sample = X.iloc[[0]].copy()
    c = st.columns([1, 1, 1, 2])
    if c[0].button("🎲 Random benign case", width="stretch"):
        ss.sample = X[y == 0].sample(1).copy(); st.rerun()
    if c[1].button("🎲 Random malignant case", width="stretch"):
        ss.sample = X[y == 1].sample(1).copy(); st.rerun()
    if c[2].button("📊 Median patient", width="stretch"):
        ss.sample = X.median().to_frame().T; st.rerun()
    model_name = c[3].selectbox("Model", list(MODELS), index=list(MODELS).index(best), label_visibility="collapsed")
    row = ss.sample.copy()
    with st.form("dx"):
        p1, p2, p3 = st.columns(3)
        patient = p1.text_input("Patient ID / name", "PT-" + dt.datetime.now().strftime("%H%M%S"))
        age = p2.number_input("Age", 18, 100, 48)
        notes = p3.text_input("Clinical notes", "")
        st.markdown("**Cell-nucleus measurements (mean values)** — ‘worst’ and SE features scale proportionally")
        cols = st.columns(5)
        new = {}
        for i, f in enumerate(MEAN_FEATS):
            lo, hi = float(X[f].min()), float(X[f].max())
            new[f] = cols[i % 5].slider(f.replace("mean ", "").title(), lo, hi, float(row[f].iloc[0]), (hi - lo) / 200, format="%.4f" if hi < 1 else "%.1f")
        go_ = st.form_submit_button("🔬 Analyse", type="primary", width="stretch")
    for f, v in new.items():
        ratio = v / (row[f].iloc[0] + 1e-9)
        row[f] = v
        for prefix in ("worst ", ):
            w = f.replace("mean ", prefix)
            row[w] = row[w] * ratio
    p = float(fitted[model_name].predict_proba(row[X.columns])[0, 1])
    label, color = risk_band(p)
    l, r = st.columns([1, 1.4])
    with l:
        g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"}, title={"text": "Malignancy probability"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": {"red": "#ef4444", "amber": "#f59e0b", "green": "#10b981"}[color]},
                                          "steps": [{"range": [0, 20], "color": "rgba(16,185,129,.15)"}, {"range": [20, 50], "color": "rgba(245,158,11,.12)"}, {"range": [50, 100], "color": "rgba(239,68,68,.15)"}]}))
        g.update_layout(height=290)
        st.plotly_chart(g, width="stretch")
        st.markdown(f"<div class='card' style='text-align:center'><span class='badge {color}' style='font-size:1rem;padding:8px 16px'>{label}</span><p style='margin-top:8px;color:var(--muted)'>{model_name} · CV AUC {res[model_name]['auc']:.3f} · sensitivity {res[model_name]['sensitivity'] * 100:.1f}%</p></div>", unsafe_allow_html=True)
        if go_:
            save_record(uid, patient, age, p, "Malignant" if p >= 0.5 else "Benign", model_name, {k: float(v) for k, v in row.iloc[0].items()}, notes)
            auth.log("diagnosis", f"{patient}: {p:.0%}")
            st.toast(f"Saved to records: {patient}")
        report = f"""ONCOSCAN AI — DECISION-SUPPORT REPORT
Generated: {dt.datetime.now():%d %b %Y %H:%M} by {st.session_state.user['name']}
Patient: {patient}   Age: {age}
Model: {model_name} (5-fold CV AUC {res[model_name]['auc']:.3f})
Malignancy probability: {p * 100:.1f}%  →  {label}
Notes: {notes or '-'}

Key measurements:
""" + "\n".join(f"  {f:<26}{row[f].iloc[0]:.4f}" for f in MEAN_FEATS) + "\n\nThis output supports, and never replaces, clinical judgement and histopathology."
        st.download_button("⬇️ Download report", report, f"oncoscan_{patient}.txt", width="stretch")
    with r:
        contrib, b0 = explain_linear(X, y, row)
        top = contrib.head(10)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=["#ef4444" if v > 0 else "#10b981" for v in top.values]))
        fig.update_layout(title="Why? Top drivers (log-odds; red → malignant, green → benign)", height=380)
        st.plotly_chart(fig, width="stretch")
    st.markdown("#### 👥 Most similar historical biopsies")
    st.dataframe(similar_cases(X, y, row), width="stretch", hide_index=True, column_config={"similarity": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

elif page == "📋 Patient Records":
    ui.hero("Patient Records", "Your saved analyses — private to your account", "📋")
    df = list_records(uid)
    if df.empty:
        st.info("No records yet — run an analysis on the **Diagnose** page and press *Analyse*.")
    else:
        df["date"] = pd.to_datetime(df.ts, unit="s").dt.strftime("%d %b %Y %H:%M")
        k = st.columns(3)
        ui.kpi(k[0], "Records", len(df), "🗂️")
        ui.kpi(k[1], "Flagged malignant", int((df.label == "Malignant").sum()), "⚠️")
        ui.kpi(k[2], "Average risk", f"{df.prob.mean() * 100:.0f}%", "📊")
        q = st.text_input("🔍 Search patient / notes")
        view = df[df.patient.str.contains(q, case=False) | df.notes.str.contains(q, case=False)] if q else df
        st.dataframe(view[["id", "date", "patient", "age", "prob", "label", "model", "notes"]], width="stretch", hide_index=True,
                     column_config={"prob": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f")})
        c1, c2 = st.columns(2)
        rid = c1.selectbox("Delete record", view.id, format_func=lambda i: f"#{i} · {df.set_index('id').patient[i]}")
        if c1.button("🗑️ Delete", width="stretch"):
            delete_record(uid, int(rid)); st.rerun()
        c2.download_button("⬇️ Export CSV", view.drop(columns="ts").to_csv(index=False), "oncoscan_records.csv", width="stretch")

elif page == "🧪 Model Arena":
    ui.hero("Model Arena", "Six classifiers, 5-fold stratified cross-validation on 569 biopsies", "🧪")
    t = pd.DataFrame({n: {k: r[k] for k in ["auc", "accuracy", "sensitivity", "specificity", "precision"]} for n, r in res.items()}).T.sort_values("auc", ascending=False)
    st.dataframe(t.style.format("{:.3f}").background_gradient(cmap="Greens"), width="stretch")
    st.markdown(f"🏆 **{best}** has the highest AUC. In cancer screening, **sensitivity** (catching malignant cases) matters most.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for n, r in res.items():
        fpr, tpr, _ = r["roc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} ({r['auc']:.3f})", line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(title="ROC curves (cross-validated)", height=420, xaxis_title="1 − specificity", yaxis_title="sensitivity", legend=dict(orientation="h", y=-0.3))
    c1.plotly_chart(fig, width="stretch")
    with c2:
        name = st.selectbox("Model", list(res), index=list(res).index(best))
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, help="Lower threshold → higher sensitivity, more false alarms")
        p = res[name]["proba"]
        pred = (p >= thr).astype(int)
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum(); tp = ((pred == 1) & (y == 1)).sum()
        st.plotly_chart(px.imshow([[tn, fp], [fn, tp]], text_auto=True, x=["Pred benign", "Pred malignant"], y=["Benign", "Malignant"], color_continuous_scale="Greens", title=f"Confusion matrix @ {thr:.2f}", height=300), width="stretch")
        m = st.columns(3)
        m[0].metric("Sensitivity", f"{tp / (tp + fn) * 100:.1f}%")
        m[1].metric("Specificity", f"{tn / (tn + fp) * 100:.1f}%")
        m[2].metric("Missed cancers", int(fn))
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(15)
    st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Top-15 features (Random Forest importance)", height=440), width="stretch")

elif page == "🔬 Data Explorer":
    ui.hero("Data Explorer", "Breast Cancer Wisconsin (Diagnostic) — 569 biopsies × 30 features", "🔬")
    k = st.columns(4)
    ui.kpi(k[0], "Samples", len(X), "🧫")
    ui.kpi(k[1], "Malignant", int(y.sum()), "🔴")
    ui.kpi(k[2], "Benign", int((y == 0).sum()), "🟢")
    ui.kpi(k[3], "Features", X.shape[1], "📐")
    lab = np.where(y == 1, "Malignant", "Benign")
    tabs = st.tabs(["📈 Distributions", "🧊 3-D PCA", "🔥 Correlations", "↔️ Scatter"])
    with tabs[0]:
        f = st.selectbox("Feature", list(X.columns), index=7)
        st.plotly_chart(px.histogram(X.assign(diagnosis=lab), x=f, color="diagnosis", marginal="box", barmode="overlay", opacity=0.7, color_discrete_map={"Malignant": "#ef4444", "Benign": "#10b981"}), width="stretch")
    with tabs[1]:
        Z = pca(X)
        st.plotly_chart(px.scatter_3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], color=lab, opacity=0.75, height=560, color_discrete_map={"Malignant": "#ef4444", "Benign": "#10b981"}, labels={"x": "PC1", "y": "PC2", "z": "PC3"}), width="stretch")
    with tabs[2]:
        st.plotly_chart(px.imshow(X[MEAN_FEATS].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=560), width="stretch")
    with tabs[3]:
        c = st.columns(2)
        fx = c[0].selectbox("X", MEAN_FEATS, index=0)
        fy = c[1].selectbox("Y", MEAN_FEATS, index=7)
        st.plotly_chart(px.scatter(X, x=fx, y=fy, color=lab, opacity=0.75, color_discrete_map={"Malignant": "#ef4444", "Benign": "#10b981"}), width="stretch")

elif page == "📁 Batch Screening":
    ui.hero("Batch Screening", "Score many biopsies at once from a CSV", "📁")
    tmpl = X.sample(25, random_state=1)
    st.download_button("⬇️ Download template (25 sample rows)", tmpl.to_csv(index=False), "oncoscan_template.csv")
    up = st.file_uploader("Upload CSV with the 30 feature columns", type="csv")
    data_ = pd.read_csv(up) if up else (tmpl if st.toggle("Use template data", True) else None)
    if data_ is not None:
        miss = [c for c in X.columns if c not in data_.columns]
        if miss:
            st.error(f"Missing columns: {miss[:5]}…")
        else:
            name = st.selectbox("Model", list(MODELS), index=list(MODELS).index(best))
            out = data_.copy()
            out.insert(0, "malignancy_prob", fitted[name].predict_proba(data_[X.columns])[:, 1].round(4))
            out.insert(1, "assessment", [risk_band(v)[0] for v in out.malignancy_prob])
            st.plotly_chart(px.histogram(out, x="malignancy_prob", nbins=20, title="Risk distribution"), width="stretch")
            st.dataframe(out.iloc[:, :6], width="stretch", hide_index=True)
            st.download_button("⬇️ Download results", out.to_csv(index=False), "oncoscan_batch_results.csv", type="primary")
else:
    ui.account_page()
