import os
import requests
import numpy as np
import streamlit as st
from gensim.models import KeyedVectors

st.set_page_config(page_title="Aide Cémantix", page_icon="🧠", layout="centered")

MODELES = {
    "Standard (Rapide - 200d CBOW - Miroir GitHub)": {
        "url": "https://github.com",
        "fichier": "frWac_200_cbow.bin",
        "limit": 150000
    },
    "Précision (Recommandé - 500d Skip-gram - Serveur Fauconnier)": {
        "url": "https://embeddings.net",
        "fichier": "frWac_500_skip.bin",
        "limit": 100000
    }
}

def fichier_valide(chemin):
    if not os.path.exists(chemin) or os.path.getsize(chemin) < 50_000_000:
        return False
    try:
        with open(chemin, "rb") as f:
            premiere_ligne = f.readline().decode("utf-8", errors="ignore").strip()
            parts = premiere_ligne.split()
            return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    except Exception:
        return False

@st.cache_resource
def charger_modele(choix_modele):
    info = MODELES[choix_modele]
    nom_fichier = info["fichier"]

    if not fichier_valide(nom_fichier):
        if os.path.exists(nom_fichier):
            os.remove(nom_fichier)
        
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(info["url"], headers=headers, stream=True, allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(nom_fichier, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

    if not fichier_valide(nom_fichier):
        raise ValueError("Le fichier téléchargé est corrompu ou n'est pas un binaire Word2Vec.")

    kv = KeyedVectors.load_word2vec_format(
        nom_fichier, 
        binary=True, 
        limit=info["limit"], 
        unicode_errors="ignore"
    )
    kv.fill_norms()
    return kv

st.title("🧠 Aide Cémantix")

choix_modele = st.selectbox("Modèle linguistique :", list(MODELES.keys()), index=1)

with st.spinner("Chargement du dictionnaire IA (peut prendre 30s au premier lancement)..."):
    try:
        modele = charger_modele(choix_modele)
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.info("Si le serveur Fauconnier est lent ou indisponible, sélectionnez le modèle **Standard (Miroir GitHub)** dans le menu ci-dessus.")
        st.stop()

if "mots_saisis" not in st.session_state:
    st.session_state.mots_saisis = []

# Ajout rapide par lot (copier-coller)
with st.expander("📋 Coller plusieurs indices d'un coup"):
    texte_lot = st.text_area(
        "Format : mot : score (un par ligne)", 
        placeholder="nature : -2.32\nformulaire : 9.81\nlisible : 7.73"
    )
    if st.button("Importer la liste"):
        ajoutes = 0
        for ligne in texte_lot.splitlines():
            if ":" in ligne:
                parties = ligne.split(":")
                m = parties[0].strip().lower()
                s_str = parties[1].replace("°C", "").replace(",", ".").strip()
                try:
                    s = float(s_str)
                    if m in modele:
                        st.session_state.mots_saisis = [x for x in st.session_state.mots_saisis if x["mot"] != m]
                        st.session_state.mots_saisis.append({"mot": m, "score": s / 100.0})
                        ajoutes += 1
                except ValueError:
                    pass
        if ajoutes > 0:
            st.rerun()

with st.form("ajout_mot", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        mot = st.text_input("Mot testé sur Cémantix", placeholder="Ex: maison").strip().lower()
    with col2:
        score = st.number_input("Score (°C)", min_value=-100.0, max_value=100.0, value=10.0, step=0.01)
    
    if st.form_submit_button("Ajouter l'indice") and mot:
        if mot in modele:
            st.session_state.mots_saisis = [m for m in st.session_state.mots_saisis if m["mot"] != mot]
            st.session_state.mots_saisis.append({"mot": mot, "score": score / 100.0})
        else:
            st.warning(f"Le mot **'{mot}'** est absent du dictionnaire.")

if st.session_state.mots_saisis:
    st.subheader("Vos indices")
    for item in st.session_state.mots_saisis:
        st.write(f"- **{item['mot']}** : {item['score'] * 100:.2f} °C")

    if st.button("🗑️ Réinitialiser"):
        st.session_state.mots_saisis = []
        st.rerun()

    st.divider()
    st.subheader("🔍 Top 30 des meilleurs candidats (Classement par proximité)")

    vecteurs_norm = modele.get_normed_vectors()
    erreur_quadratique = np.zeros(len(modele.index_to_key))

    for item in st.session_state.mots_saisis:
        idx_mot = modele.key_to_index[item["mot"]]
        vec_mot = vecteurs_norm[idx_mot]
        
        similarites = np.dot(vecteurs_norm, vec_mot)
        poids = 1.0 + max(0.0, item["score"] * 5.0)
        erreur_quadratique += poids * ((similarites - item["score"]) ** 2)

    indices_tries = np.argsort(erreur_quadratique)
    mots_deja_testes = {m["mot"] for m in st.session_state.mots_saisis}
    
    resultats = []
    for idx in indices_tries:
        candidat = modele.index_to_key[idx]
        if (
            candidat not in mots_deja_testes 
            and candidat.isalpha() 
            and candidat.islower() 
            and len(candidat) >= 3
        ):
            ecart_moyen = np.sqrt(erreur_quadratique[idx] / len(st.session_state.mots_saisis)) * 100
            resultats.append(f"**{candidat}** (±{ecart_moyen:.1f}°)")
        if len(resultats) >= 30:
            break
    
    st.write(", ".join(resultats))
