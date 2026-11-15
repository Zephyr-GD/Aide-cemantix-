import os
import random
import urllib.request
from gensim.models import KeyedVectors
import numpy as np
import streamlit as st

st.set_page_config(page_title="Aide Cémantix", page_icon="🧠", layout="centered")


@st.cache_resource
def charger_modele():
    nom_fichier = "frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
    url = "https://embeddings.net"

    if not os.path.exists(nom_fichier):
        urllib.request.urlretrieve(url, nom_fichier)

    return KeyedVectors.load_word2vec_format(nom_fichier, binary=True)


st.title("🧠 Aide Cémantix")

with st.spinner("Chargement du dictionnaire IA..."):
    modele = charger_modele()

if "mots_saisis" not in st.session_state:
    st.session_state.mots_saisis = []

# --- INTERFACE [Case mot] Mot [Case %] % ---
col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    nouveau_mot = st.text_input(
        "Mot", placeholder="Mot", label_visibility="collapsed"
    )
with col2:
    nouveau_score = st.number_input(
        "%",
        min_value=-100.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        label_visibility="collapsed",
    )
with col3:
    ajouter = st.button("➕")

if ajouter and nouveau_mot:
    mot_clean = nouveau_mot.strip().lower()
    if mot_clean in modele:
        st.session_state.mots_saisis.append((mot_clean, nouveau_score))
    else:
        st.error(f"'{mot_clean}' inconnu.")

# --- LISTE DES MOTS SAISIS ---
if st.session_state.mots_saisis:
    for idx, (m, s) in enumerate(st.session_state.mots_saisis):
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(f"**{m}**")
        c2.write(f"{s} %")
        if c3.button("❌", key=f"del_{idx}"):
            st.session_state.mots_saisis.pop(idx)
            st.rerun()

    st.divider()

    # --- CHOIX DE L'INTERVALLE ET RECHERCHE ---
    intervalle = st.slider(
        "Intervalle (Top)", min_value=1, max_value=1000, value=(900, 1000)
    )

    if st.button("🔍 Trouver un mot", type="primary", use_container_width=True):
        vecteurs = []
        mots_deja_joues = []

        for mot, score in st.session_state.mots_saisis:
            mots_deja_joues.append(mot)
            poids = score if score > 0 else 0.1
            vecteurs.append(modele[mot] * poids)

        vecteur_cible = np.mean(vecteurs, axis=0)
        voisins = modele.similar_by_vector(
            vecteur_cible, topn=intervalle[1] + 50
        )

        candidats = [
            m.lower()
            for m, _ in voisins
            if m.lower() not in mots_deja_joues and m.isalpha()
        ]

        tranche = candidats[intervalle[0] : intervalle[1]]

        if tranche:
            st.success(f"💡 Essaie : **{random.choice(tranche).upper()}**")
        else:
            st.warning("Aucun mot trouvé dans cet intervalle.")
      
