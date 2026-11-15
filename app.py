import os
import requests
import streamlit as st
from gensim.models import KeyedVectors

@st.cache_resource
def charger_modele():
    """Télécharge et charge le modèle Word2Vec français depuis GitHub Releases."""
    url = "https://github.com/lovasoa/cemantriche/releases/download/v0/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
    nom_fichier = "frWac_200_cbow_cut100.bin"

    # Vérifie que le fichier n'existe pas ou qu'il n'est pas vide (erreur précédente)
    if not os.path.exists(nom_fichier) or os.path.getsize(nom_fichier) < 10_000_000:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(nom_fichier, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # limit=150000 évite de dépasser la limite de 1 Go de RAM sur Streamlit Cloud
    return KeyedVectors.load_word2vec_format(nom_fichier, binary=True, limit=150000)
    

st.title("🧠 Aide Cémantix")
st.write("Trouvez le mot secret en croisant les scores de similarité de vos essais.")

# Chargement du modèle
with st.spinner("Chargement du dictionnaire IA (peut prendre 10 à 20 secondes au démarrage)..."):
    try:
        modele = charger_modele()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

# Initialisation de l'historique des mots
if "mots_saisis" not in st.session_state:
    st.session_state.mots_saisis = []

# Formulaire pour ajouter un mot et son score
with st.form("ajout_mot", clear_on_submit=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        mot = st.text_input("Mot testé sur Cémantix", placeholder="Ex: maison").strip().lower()
    with col2:
        # Cémantix affiche un score entre -100 et 100 (qui correspond à la similarité cosinus * 100)
        score = st.number_input("Score (°C)", min_value=-100.0, max_value=100.0, value=25.0, step=0.01)
    
    soumis = st.form_submit_button("Ajouter l'indice")

if soumis and mot:
    if mot in modele:
        # Évite les doublons en mettant à jour si le mot existe déjà
        st.session_state.mots_saisis = [m for m in st.session_state.mots_saisis if m["mot"] != mot]
        st.session_state.mots_saisis.append({"mot": mot, "score": score / 100.0})
    else:
        st.warning(f"Le mot **'{mot}'** n'est pas dans le dictionnaire du modèle.")

# Affichage des indices enregistrés et recherche des candidats
if st.session_state.mots_saisis:
    st.subheader("Vos indices")
    for item in st.session_state.mots_saisis:
        st.write(f"- **{item['mot']}** : {item['score'] * 100:.2f} °C")

    if st.button("🗑️ Réinitialiser les indices"):
        st.session_state.mots_saisis = []
        st.rerun()

    st.divider()
    st.subheader("🔍 Mots candidats les plus probables")

    # Paramètre de tolérance pour l'intervalle de recherche
    marge = st.slider(
        "Marge d'erreur tolérée sur le score (± °C)", 
        min_value=1.0, max_value=15.0, value=4.0, step=0.5
    ) / 100.0

    # Calcul vectorisé des similarités sur tout le vocabulaire
    vecteurs_vocab = modele.vectors
    normes_vocab = np.linalg.norm(vecteurs_vocab, axis=1)
    
    # Masque booléen pour filtrer les mots qui respectent TOUS les indices
    masque_valide = np.ones(len(modele.index_to_key), dtype=bool)
    erreur_totale = np.zeros(len(modele.index_to_key))

    for item in st.session_state.mots_saisis:
        vec_mot = modele[item["mot"]]
        norme_mot = np.linalg.norm(vec_mot)
        
        # Similarité cosinus avec tous les mots du vocabulaire
        similarites = np.dot(vecteurs_vocab, vec_mot) / (normes_vocab * norme_mot)
        diff = np.abs(similarites - item["score"])
        
        masque_valide &= (diff <= marge)
        erreur_totale += diff

    indices_candidats = np.where(masque_valide)[0]

    if len(indices_candidats) == 0:
        st.info("Aucun mot trouvé dans cet intervalle. Essayez d'augmenter la **marge d'erreur** ci-dessus.")
    else:
        # Tri des candidats du plus proche au moins proche
        indices_tries = indices_candidats[np.argsort(erreur_totale[indices_candidats])]
        mots_deja_testes = {m["mot"] for m in st.session_state.mots_saisis}
        
        resultats = []
        for idx in indices_tries:
            candidat = modele.index_to_key[idx]
            # On filtre les mots déjà testés et ceux contenant des chiffres ou caractères spéciaux
            if candidat not in mots_deja_testes and candidat.isalpha():
                resultats.append(candidat)
            if len(resultats) >= 30:
                break
        
        if resultats:
            st.success(f"**{len(indices_candidats)}** mots correspondent. Voici les meilleurs candidats :")
            st.write(", ".join([f"**{m}**" for m in resultats]))
        else:
            st.info("Tous les mots correspondants ont déjà été testés.")
            
