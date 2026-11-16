import os
import numpy as np
import streamlit as st
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Aide Cémantix", page_icon="🧠", layout="centered")

@st.cache_resource
def charger_modele():
    chemin_fichier = hf_hub_download(
        repo_id="Zephyr0001/Cemantix",
        filename="frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin",
        repo_type="dataset"
    )
    
    kv = KeyedVectors.load_word2vec_format(
        chemin_fichier, 
        binary=True, 
        unicode_errors="ignore"
    )
    kv.fill_norms()
    
    if os.path.exists(chemin_fichier):
        os.remove(chemin_fichier)
        
    return kv

st.title("🧠 Aide Cémantix")

with st.spinner("Chargement du dictionnaire IA (quelques secondes au démarrage)..."):
    try:
        modele = charger_modele()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

if "mots_saisis" not in st.session_state:
    st.session_state.mots_saisis = []

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
    st.subheader("🔍 Top 30 des meilleurs candidats (Classement optimisé)")

    vecteurs_norm = modele.get_normed_vectors()
    # On limite la recherche aux 45 000 mots les plus fréquents (vocabulaire réel de Cémantix)
    limite_vocab = min(45000, len(modele.index_to_key))
    vecteurs_courants = vecteurs_norm[:limite_vocab]
    
    erreur_quadratique = np.zeros(limite_vocab)

    for item in st.session_state.mots_saisis:
        idx_mot = modele.key_to_index[item["mot"]]
        vec_mot = vecteurs_norm[idx_mot]
        
        similarites = np.dot(vecteurs_courants, vec_mot)
        
        # Correction de l'échelle Cémantix : 67°C sur Cémantix ≈ 0.50 de similarité brute
        score_brut = item["score"]
        if score_brut > 0.35:
            cible_ajustee = 0.35 + (score_brut - 0.35) * 0.55
        else:
            cible_ajustee = score_brut

        # Poids exponentiel pour que les mots > 50°C guident la recherche
        poids = np.exp(score_brut * 6.0) if score_brut > 0 else 0.5
        
        erreur_quadratique += poids * ((similarites - cible_ajustee) ** 2)

    indices_tries = np.argsort(erreur_quadratique)
    mots_deja_testes = {m["mot"] for m in st.session_state.mots_saisis}
    
    # Exclure les terminaisons de verbes conjugués
    terminaisons_exclues = ("ait", "aient", "era", "eront", "iez", "ions", "âmes", "âtes", "èrent")
    
    resultats = []
    for idx in indices_tries:
        candidat = modele.index_to_key[idx]
        if (
            candidat not in mots_deja_testes 
            and candidat.isalpha() 
            and candidat.islower() 
            and len(candidat) >= 3
            and not candidat.endswith(terminaisons_exclues)
        ):
            resultats.append(f"**{candidat}**")
        if len(resultats) >= 30:
            break
    
    st.write(", ".join(resultats))
