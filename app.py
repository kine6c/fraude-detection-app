
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="centered"
)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    model     = joblib.load("model/fraud_model.pkl")
    scaler    = joblib.load("model/scaler.pkl")
    le_target = joblib.load("model/le_target.pkl")
    le_type   = joblib.load("model/le_type.pkl")
    le_status = joblib.load("model/le_status.pkl")
    le_loc    = joblib.load("model/le_loc.pkl")
    return model, scaler, le_target, le_type, le_status, le_loc

model, scaler, le_target, le_type, le_status, le_loc = load_model()

# --- Titre ---
st.title("🏦 Détection de Fraude Bancaire")
st.markdown("Analysez une transaction et détectez si elle est **Normale**, **Suspecte** ou **Frauduleuse**.")
st.divider()

# --- Saisie manuelle ---
st.subheader("📝 Saisir une transaction")

col1, col2 = st.columns(2)

with col1:
    montant = st.number_input("Montant (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
    type_transaction = st.selectbox("Type de transaction", le_type.classes_)

with col2:
    status = st.selectbox("Status opération", le_status.classes_)
    localisation = st.selectbox("Localisation", le_loc.classes_)

if st.button("🔍 Analyser la transaction", type="primary"):
    # Encodage
    type_enc   = le_type.transform([type_transaction])[0]
    status_enc = le_status.transform([status])[0]
    loc_enc    = le_loc.transform([localisation])[0]

    features = np.array([[type_enc, status_enc, loc_enc, montant]])
    features_sc = scaler.transform(features)

    prediction = model.predict(features_sc)[0]
    proba = model.predict_proba(features_sc)[0]
    label = le_target.inverse_transform([prediction])[0]

    st.divider()
    if label == "Fraude":
        st.error(f"🚨 Transaction FRAUDULEUSE détectée ! (confiance : {max(proba)*100:.1f}%)")
    elif label == "Suspect":
        st.warning(f"⚠️ Transaction SUSPECTE (confiance : {max(proba)*100:.1f}%)")
    else:
        st.success(f"✅ Transaction NORMALE (confiance : {max(proba)*100:.1f}%)")

    # Probabilités par classe
    st.subheader("Probabilités par classe")
    proba_df = pd.DataFrame({"Classe": le_target.classes_, "Probabilité (%)": (proba*100).round(2)})
    st.dataframe(proba_df, hide_index=True)

st.divider()

# --- Upload CSV ---
st.subheader("📂 Analyser un fichier CSV de transactions")
fichier = st.file_uploader("Déposez un fichier CSV", type=["csv"])

if fichier is not None:
    df_upload = pd.read_csv(fichier, sep=";")
    st.write("Aperçu :", df_upload.head())

    if st.button("🚀 Lancer l'analyse du lot"):
        df_upload["Type_enc"]   = le_type.transform(df_upload["Type de transaction"])
        df_upload["Status_enc"] = le_status.transform(df_upload["Status operation"])
        df_upload["Loc_enc"]    = le_loc.transform(df_upload["Localisation"])

        X_lot = df_upload[["Type_enc", "Status_enc", "Loc_enc", "Montant"]].values
        X_lot_sc = scaler.transform(X_lot)

        df_upload["Prédiction"] = le_target.inverse_transform(model.predict(X_lot_sc))
        df_upload["Proba Fraude (%)"] = (model.predict_proba(X_lot_sc)[:, list(le_target.classes_).index("Fraude")] * 100).round(2)

        nb_fraudes = (df_upload["Prédiction"] == "Fraude").sum()
        nb_suspects = (df_upload["Prédiction"] == "Suspect").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Fraudes", nb_fraudes)
        col2.metric("⚠️ Suspectes", nb_suspects)
        col3.metric("✅ Normales", len(df_upload) - nb_fraudes - nb_suspects)

        st.dataframe(df_upload[["Type de transaction", "Localisation", "Montant", "Prédiction", "Proba Fraude (%)"]])

        csv_export = df_upload.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger les résultats", csv_export, "resultats_fraude.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("-- Détection de fraude bancaire")
