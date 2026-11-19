import re
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Aide Cémantix Pro", page_icon="🧠", layout="wide")


@st.cache_resource
def charger_donnees():
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

    limite_vocab = min(45000, len(kv.index_to_key))
    vecteurs_norm = kv.get_normed_vectors()
    vecteurs_courants = vecteurs_norm[:limite_vocab]

    return kv, vecteurs_norm, vecteurs_courants, limite_vocab


def lire_url(modele):
    """Reconstruit indices et liste noire depuis les paramètres d'URL."""
    params = st.query_params
    indices = []
    for part in params.get("i", "").split(","):
        if ":" not in part:
            continue
        m, s = part.rsplit(":", 1)
        try:
            s = float(s)
        except ValueError:
            continue
        if m in modele and not any(x["mot"] == m for x in indices):
            indices.append({"mot": m, "score": s / 100.0})
    bannis = []
    for m in params.get("b", "").split(","):
        if m and m not in bannis:
            bannis.append(m)
    return indices, bannis


def ecrire_url():
    """Écrit l'état courant dans l'URL (sans relancer le script)."""
    val_i = ",".join(f"{x['mot']}:{x['score'] * 100:.2f}" for x in st.session_state.mots_saisis)
    val_b = ",".join(st.session_state.liste_noire)
    for cle, val in (("i", val_i), ("b", val_b)):
        if val:
            if st.query_params.get(cle) != val:
                st.query_params[cle] = val
        elif cle in st.query_params:
            del st.query_params[cle]


st.title("🧠 Aide Cémantix")

with st.spinner("Chargement du dictionnaire IA (quelques secondes au démarrage)..."):
    try:
        modele, vecteurs_norm, vecteurs_courants, limite_vocab = charger_donnees()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

# --- ÉTAT DE SESSION (restauré depuis l'URL au premier chargement) ---
if "mots_saisis" not in st.session_state:
    indices_url, bannis_url = lire_url(modele)
    st.session_state.mots_saisis = indices_url
    st.session_state.liste_noire = bannis_url

# Synchronise l'URL avec l'état à chaque exécution
ecrire_url()

# --- BARRE LATÉRALE : PARAMÈTRES DE RECHERCHE ---
with st.sidebar:
    st.header("⚙️ Réglages de l'algorithme")

    diversite = st.slider(
        "Diversité des suggestions",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="0 = mots les plus proches possibles (souvent des synonymes). Plus la valeur est haute, plus les idées sont variées."
    )

    exclure_pluriels = st.checkbox(
        "Exclure les pluriels (-s, -x)",
        value=True,
        help="Cémantix n'utilise généralement que le singulier. Décochez si vous cherchez un mot comme 'souris' ou 'temps'."
    )

    exclure_adverbes = st.checkbox(
        "Exclure les adverbes (-ment)",
        value=True
    )

    st.divider()
    st.subheader("🔮 Simulateur de mot")
    mot_test = st.text_input("Tester un mot avant de le jouer :", placeholder="Ex: forêt").strip().lower()
    if mot_test:
        if mot_test not in modele:
            st.warning("Mot absent du dictionnaire.")
        elif not st.session_state.mots_saisis:
            st.info("Ajoutez d'abord des indices pour estimer son score.")
        else:
            vec_test = vecteurs_norm[modele.key_to_index[mot_test]]
            poids_total = 0.0
            score_estime = 0.0
            for item in st.session_state.mots_saisis:
                vec_ind = vecteurs_norm[modele.key_to_index[item["mot"]]]
                sim = np.dot(vec_test, vec_ind)
                p = (max(0.0, item["score"]) + 0.2) ** 3
                score_estime += sim * p
                poids_total += p

            proximite_globale = (score_estime / poids_total) * 100
            st.metric(label=f"Proximité estimée pour '{mot_test}'", value=f"{proximite_globale:.1f} %")

    st.divider()
    st.subheader(f"🚫 Liste noire ({len(st.session_state.liste_noire)})")

    with st.form("form_liste_noire", clear_on_submit=True):
        saisie_bannis = st.text_input(
            "Bannir un ou plusieurs mots :",
            placeholder="Ex: pluie, vent"
        )
        if st.form_submit_button("Bannir", use_container_width=True):
            for m in re.split(r"[,;\s]+", saisie_bannis.strip().lower()):
                if m and m not in st.session_state.liste_noire:
                    st.session_state.liste_noire.append(m)
            st.rerun()

    if st.session_state.liste_noire:
        for m in sorted(st.session_state.liste_noire):
            c_m, c_b = st.columns([4, 1])
            c_m.markdown(f"~~{m}~~")
            if c_b.button("↩️", key=f"unban_{m}", help=f"Retirer '{m}' de la liste noire"):
                st.session_state.liste_noire.remove(m)
                st.rerun()
        if st.button("Vider la liste noire", use_container_width=True):
            st.session_state.liste_noire = []
            st.rerun()
    else:
        st.caption("Aucun mot banni.")

    st.divider()
    st.caption("💾 Votre partie (indices + liste noire) est sauvegardée dans l'URL. "
               "Mettez la page en favori ou copiez le lien pour la retrouver ou la partager.")

# --- ZONE PRINCIPALE : SAISIE DES INDICES ---
col_saisie, col_lot = st.columns([1, 1])

with col_saisie:
    with st.form("ajout_mot", clear_on_submit=True):
        st.subheader("➕ Ajouter un indice")
        c1, c2 = st.columns([2, 1])
        with c1:
            mot = st.text_input("Mot testé", placeholder="Ex: maison").strip().lower()
        with c2:
            score = st.number_input("Score (°C)", min_value=-100.0, max_value=100.0, value=15.0, step=0.01)

        if st.form_submit_button("Ajouter", use_container_width=True) and mot:
            if mot in modele:
                st.session_state.mots_saisis = [m for m in st.session_state.mots_saisis if m["mot"] != mot]
                st.session_state.mots_saisis.append({"mot": mot, "score": score / 100.0})
                st.rerun()
            else:
                st.warning(f"Le mot **'{mot}'** est absent du dictionnaire.")

with col_lot:
    with st.expander("📋 Import rapide (Copier-coller depuis Cémantix)", expanded=False):
        texte_lot = st.text_area(
            "Collez directement votre tableau Cémantix ou une liste 'mot : score'",
            placeholder="124   maison   24,56 °C\nnature : -2.32\nformulaire 9.81",
            height=110
        )
        if st.button("Importer la liste", use_container_width=True):
            ajoutes = 0
            pattern = re.compile(r"(?:^\d+\s+)?([a-zàâäéèêëîïôöùûüçœæ-]+)[\s:]+(-?\d+(?:[.,]\d+)?)", re.IGNORECASE)
            for ligne in texte_lot.splitlines():
                match = pattern.search(ligne.strip())
                if match:
                    m = match.group(1).lower()
                    s_str = match.group(2).replace(",", ".")
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
            else:
                st.warning("Aucun mot reconnu dans le texte collé.")

# --- AFFICHAGE DES INDICES SAISIS ---
if st.session_state.mots_saisis:
    st.divider()
    header_col, btn_col = st.columns([4, 1])
    with header_col:
        st.subheader(f"📌 Vos indices ({len(st.session_state.mots_saisis)})")
    with btn_col:
        if st.button("🗑️ Tout effacer", use_container_width=True):
            st.session_state.mots_saisis = []
            st.rerun()

    indices_ordonnes = sorted(st.session_state.mots_saisis, key=lambda x: x["score"], reverse=True)

    cols_indices = st.columns(4)
    for idx_item, item in enumerate(indices_ordonnes):
        with cols_indices[idx_item % 4]:
            c_txt, c_del = st.columns([4, 1])
            temp = item["score"] * 100
            emoji = "🔥" if temp >= 40 else ("☀️" if temp >= 20 else ("🌤️" if temp > 0 else "❄️"))
            c_txt.markdown(f"{emoji} **{item['mot']}** : `{temp:.2f} °C`")
            if c_del.button("❌", key=f"del_{item['mot']}", help=f"Supprimer {item['mot']}"):
                st.session_state.mots_saisis = [x for x in st.session_state.mots_saisis if x["mot"] != item["mot"]]
                st.rerun()

    st.divider()

    # --- CALCUL DES MEILLEURS CANDIDATS ---
    erreur_quadratique = np.zeros(limite_vocab)
    somme_poids = 0.0

    for item in st.session_state.mots_saisis:
        idx_mot = modele.key_to_index[item["mot"]]
        vec_mot = vecteurs_norm[idx_mot]
        similarites = np.dot(vecteurs_courants, vec_mot)

        score_brut = item["score"]
        if score_brut > 0.35:
            cible_ajustee = 0.35 + (score_brut - 0.35) * 0.55
        else:
            cible_ajustee = score_brut

        poids = (max(0.0, score_brut) + 0.2) ** 3
        erreur_quadratique += poids * ((similarites - cible_ajustee) ** 2)
        somme_poids += poids

    indices_tries = np.argsort(erreur_quadratique)

    mots_deja_testes = {m["mot"] for m in st.session_state.mots_saisis}
    mots_bannis = set(st.session_state.liste_noire)
    terminaisons_exclues = ("ait", "aient", "era", "eront", "iez", "ions", "âmes", "âtes", "èrent")

    pool_candidats = []
    for idx in indices_tries:
        candidat = modele.index_to_key[idx]
        if (
            candidat not in mots_deja_testes
            and candidat not in mots_bannis
            and candidat.isalpha()
            and candidat.islower()
            and len(candidat) >= 3
            and not candidat.endswith(terminaisons_exclues)
            and (not exclure_pluriels or not candidat.endswith(("s", "x")))
            and (not exclure_adverbes or not candidat.endswith("ment"))
        ):
            rmse = np.sqrt(erreur_quadratique[idx] / somme_poids)
            correspondance = max(0.0, min(100.0, (1.0 - rmse * 2.5) * 100))
            pool_candidats.append((idx, candidat, correspondance))
        if len(pool_candidats) >= 150:
            break

    # Sélection finale de 30 mots avec diversité (MMR vectorisé)
    resultats = []
    if pool_candidats:
        mat_pool = vecteurs_courants[[c[0] for c in pool_candidats]]
        corr_pool = np.array([c[2] for c in pool_candidats])
        sim_max = np.full(len(pool_candidats), -1.0)
        disponibles = np.ones(len(pool_candidats), dtype=bool)

        for n in range(min(30, len(pool_candidats))):
            penalite = diversite * sim_max * 100 if n > 0 else 0.0
            scores_mmr = np.where(disponibles, corr_pool - penalite, -np.inf)
            meilleur = int(np.argmax(scores_mmr))
            resultats.append(pool_candidats[meilleur])
            disponibles[meilleur] = False
            sim_max = np.maximum(sim_max, mat_pool @ mat_pool[meilleur])

    # --- ONGLETS : SUGGESTIONS VS CARTE 2D ---
    tab_suggestions, tab_carte = st.tabs(["🔍 Top 30 des suggestions", "🗺️ Carte sémantique 2D"])

    with tab_suggestions:
        colonnes = st.columns(3)
        for i, (_, mot_candidat, pct) in enumerate(resultats):
            with colonnes[i % 3]:
                c_mot, c_pct, c_add, c_ban = st.columns([4, 2, 1, 1])
                c_mot.markdown(f"**{i + 1}.** `{mot_candidat}`")
                c_pct.markdown(f"**{pct:.1f}%**")

                with c_add.popover("➕", help=f"Jouer '{mot_candidat}' et saisir son score"):
                    st.markdown(f"**{mot_candidat}**")
                    sc = st.number_input(
                        "Score (°C)", min_value=-100.0, max_value=100.0,
                        value=15.0, step=0.01, key=f"sc_{mot_candidat}"
                    )
                    if st.button("Valider", key=f"ok_{mot_candidat}", use_container_width=True):
                        st.session_state.mots_saisis = [
                            x for x in st.session_state.mots_saisis if x["mot"] != mot_candidat
                        ]
                        st.session_state.mots_saisis.append({"mot": mot_candidat, "score": sc / 100.0})
                        st.rerun()

                if c_ban.button("🚫", key=f"ban_{mot_candidat}", help=f"Bannir '{mot_candidat}'"):
                    if mot_candidat not in st.session_state.liste_noire:
                        st.session_state.liste_noire.append(mot_candidat)
                    st.rerun()

                st.progress(int(pct))

    with tab_carte:
        if len(st.session_state.mots_saisis) < 3:
            st.info("Ajoutez au moins 3 indices pour générer la carte sémantique 2D.")
        else:
            st.caption("Visualisation de la proximité entre vos indices et le Top 10 des suggestions.")
            mots_carte = []
            vecteurs_carte = []
            types_carte = []
            tailles_carte = []

            for item in st.session_state.mots_saisis:
                mots_carte.append(f"{item['mot']} ({item['score']*100:.0f}°C)")
                vecteurs_carte.append(vecteurs_norm[modele.key_to_index[item["mot"]]])
                types_carte.append("Indice testé")
                tailles_carte.append(max(8, (item["score"] + 1) * 12))

            for _, mot_cand, pct in resultats[:10]:
                mots_carte.append(f"💡 {mot_cand}")
                vecteurs_carte.append(vecteurs_norm[modele.key_to_index[mot_cand]])
                types_carte.append("Top 10 Suggestion")
                tailles_carte.append(12)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(np.array(vecteurs_carte))

            fig = px.scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                text=mots_carte,
                color=types_carte,
                size=tailles_carte,
                color_discrete_map={"Indice testé": "#FF4B4B", "Top 10 Suggestion": "#00CC96"},
                labels={"x": "Axe1", "y": "Axe2", "color": "Catégorie"}
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=550, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
