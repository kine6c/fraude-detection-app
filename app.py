import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Configuration de la page ---
st.set_page_config(
    st.markdown("# 🏦 Détection de Fraude Bancaire"),
    st.markdown("---"),
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

# --- Sidebar ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", [
    "📝 Saisir une transaction",
    "📂 Analyser un fichier CSV",
    "📊 Données"
])
st.sidebar.markdown("---")
st.sidebar.caption("Détection de fraude bancaire")


# ══════════════════════════════════════════════
# PAGE 1 : SAISIR UNE TRANSACTION
# ══════════════════════════════════════════════
if page == "📝 Saisir une transaction":

    st.title("📝 Saisir une transaction")
    st.markdown("Renseignez les informations de la transaction pour détecter si elle est **Normale**, **Suspecte** ou **Frauduleuse**.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        montant          = st.number_input("Montant (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
        type_transaction = st.selectbox("Type de transaction", le_type.classes_)

    with col2:
        status       = st.selectbox("Status opération", le_status.classes_)
        localisation = st.selectbox("Localisation", le_loc.classes_)

    if st.button("🔍 Analyser la transaction", type="primary"):
        type_enc   = le_type.transform([type_transaction])[0]
        status_enc = le_status.transform([status])[0]
        loc_enc    = le_loc.transform([localisation])[0]

        features    = np.array([[type_enc, status_enc, loc_enc, montant]])
        features_sc = scaler.transform(features)

        prediction = model.predict(features_sc)[0]
        proba      = model.predict_proba(features_sc)[0]
        label      = le_target.inverse_transform([prediction])[0]

        st.divider()
        if label == "Fraude":
            st.error(f"🚨 Transaction FRAUDULEUSE ! (confiance : {max(proba)*100:.1f}%)")
        elif label == "Suspect":
            st.warning(f"⚠️ Transaction SUSPECTE (confiance : {max(proba)*100:.1f}%)")
        else:
            st.success(f"✅ Transaction NORMALE (confiance : {max(proba)*100:.1f}%)")

        st.subheader("Probabilités par classe")
        proba_df = pd.DataFrame({
            "Classe": le_target.classes_,
            "Probabilité (%)": (proba * 100).round(2)
        })
        st.dataframe(proba_df, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 2 : ANALYSER UN FICHIER CSV
# ══════════════════════════════════════════════
elif page == "📂 Analyser un fichier CSV":

    st.title("📂 Analyser un fichier CSV")
    st.markdown("Chargez un fichier CSV de transactions pour lancer une analyse en lot.")
    st.divider()

    fichier = st.file_uploader("Déposez votre fichier CSV", type=["csv"])

    if fichier is not None:
        df_upload = pd.read_csv(fichier, sep=";")
        st.write("Aperçu :", df_upload.head())

        if st.button("🚀 Lancer l'analyse du lot", type="primary"):
            df_upload["Type_enc"]   = le_type.transform(df_upload["Type de transaction"])
            df_upload["Status_enc"] = le_status.transform(df_upload["Status operation"])
            df_upload["Loc_enc"]    = le_loc.transform(df_upload["Localisation"])

            X_lot    = df_upload[["Type_enc", "Status_enc", "Loc_enc", "Montant"]].values
            X_lot_sc = scaler.transform(X_lot)

            df_upload["Prédiction"]       = le_target.inverse_transform(model.predict(X_lot_sc))
            df_upload["Proba Fraude (%)"] = (model.predict_proba(X_lot_sc)[:, list(le_target.classes_).index("Fraude")] * 100).round(2)

            # Sauvegarder pour la page Données
            st.session_state["df_result"] = df_upload

            nb_fraudes  = (df_upload["Prédiction"] == "Fraude").sum()
            nb_suspects = (df_upload["Prédiction"] == "Suspect").sum()
            nb_normaux  = len(df_upload) - nb_fraudes - nb_suspects

            c1, c2, c3 = st.columns(3)
            c1.metric("🚨 Fraudes",   nb_fraudes)
            c2.metric("⚠️ Suspectes", nb_suspects)
            c3.metric("✅ Normales",  nb_normaux)

            st.divider()
            st.dataframe(df_upload[["Type de transaction", "Localisation", "Montant", "Prédiction", "Proba Fraude (%)"]])

            csv_export = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger les résultats", csv_export, "resultats_fraude.csv", "text/csv")

            st.success("✅ Résultats disponibles dans la page 📊 Données")


# ══════════════════════════════════════════════
# PAGE 3 : DONNÉES — TABLEAU DE BORD
# ══════════════════════════════════════════════
elif page == "📊 Données":

    st.title("📊 Tableau de Bord — Transactions Bancaires")
    st.divider()

    if "df_result" not in st.session_state:
        st.warning("⬅️ Allez dans **Analyser un fichier CSV** pour charger et analyser vos données.")
        st.stop()

    df = st.session_state["df_result"]
    col_target = "Prédiction"
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    # ── KPI ──────────────────────────────────────────────────────────────
    nb_transactions = len(df)
    nb_fraudes      = (df[col_target] == "Fraude").sum()
    nb_clients      = df["ID Clients"].nunique()
    montant_median  = df["Montant"].median()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💳 Transactions",   f"{nb_transactions:,}")
    k2.metric("🚨 Fraudes",        f"{nb_fraudes:,}")
    k3.metric("👤 Clients",        f"{nb_clients:,}")
    k4.metric("💰 Montant médian", f"{montant_median:,.0f} FCFA")

    st.divider()

    # ── 1. Répartition des classes ────────────────────────────────────────
    st.subheader("Répartition des transactions")
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

    counts = df[col_target].value_counts()
    axes1[0].bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="white", width=0.5)
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        axes1[0].text(i, val + 10, f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold", fontsize=9)
    axes1[0].set_title("Nombre par classe", fontweight="bold")
    axes1[0].set_ylabel("Nombre de transactions")
    axes1[0].set_ylim(0, max(counts) * 1.2)
    axes1[0].grid(axis="y", alpha=0.3)

    axes1[1].pie(counts.values, labels=counts.index, colors=colors[:len(counts)],
                 autopct="%1.1f%%", startangle=90,
                 wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes1[1].set_title("Répartition (%)", fontweight="bold")
    st.pyplot(fig1)
    plt.close()

    # ── 2. Montant par classe ─────────────────────────────────────────────
    st.subheader("Montant des transactions par classe")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    for label, color in zip(["Normal", "Suspect", "Fraude"], colors):
        subset = df[df[col_target] == label]["Montant"]
        if len(subset) > 0:
            axes2[0].hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    axes2[0].set_title("Distribution du Montant", fontweight="bold")
    axes2[0].set_xlabel("Montant (FCFA)")
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    data_plot = [df[df[col_target] == t]["Montant"].values for t in ["Normal", "Suspect", "Fraude"] if t in df[col_target].values]
    labels_bp = [t for t in ["Normal", "Suspect", "Fraude"] if t in df[col_target].values]
    bp = axes2[1].boxplot(data_plot, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color + "99")
    axes2[1].set_xticklabels(labels_bp)
    axes2[1].set_title("Boxplot du Montant", fontweight="bold")
    axes2[1].set_ylabel("Montant (FCFA)")
    axes2[1].grid(axis="y", alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    # ── 3. Type de transaction ────────────────────────────────────────────
    st.subheader("Type de transaction par classe")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ct = pd.crosstab(df["Type de transaction"], df[col_target])
    ct.plot(kind="bar", ax=ax3, color=colors[:len(ct.columns)], edgecolor="white", width=0.6)
    ax3.set_title("Type de transaction par classe", fontweight="bold")
    ax3.set_xlabel("Type de transaction")
    ax3.set_ylabel("Nombre")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    ax3.legend(title="Classe")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3)
    plt.close()

    # ── 4. Top 10 localisations des fraudes ───────────────────────────────
    st.subheader("Top 10 localisations des fraudes")
    fraudes_loc = df[df[col_target] == "Fraude"]["Localisation"].value_counts().head(10)
    if len(fraudes_loc) > 0:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.bar(fraudes_loc.index, fraudes_loc.values, color="#e74c3c", edgecolor="white")
        ax4.set_title("Top 10 localisations des fraudes", fontweight="bold")
        ax4.set_xlabel("Localisation")
        ax4.set_ylabel("Nombre de fraudes")
        ax4.set_xticklabels(fraudes_loc.index, rotation=45, ha="right")
        ax4.grid(axis="y", alpha=0.3)
        st.pyplot(fig4)
        plt.close()
    else:
        st.info("Aucune fraude détectée dans ce fichier.")