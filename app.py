import re
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Aide Cémantix Pro v2", page_icon="🧠", layout="wide")

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

def sauvegarder_url():
    if st.session_state.mots_saisis:
        chaine = ",".join(f"{m['mot']}:{m['score']*100:.2f}" for m in st.session_state.mots_saisis)
        st.query_params["indices"] = chaine
    else:
        st.query_params.clear()

def charger_depuis_url(modele):
    if "indices" in st.query_params and not st.session_state.mots_saisis:
        brut = st.query_params["indices"]
        charges = []
        for bloc in brut.split(","):
            if ":" in bloc:
                m, s = bloc.split(":", 1)
                m = m.strip().lower()
                try:
                    val = float(s)
                    if m in modele:
                        charges.append({"mot": m, "score": val / 100.0})
                except ValueError:
                    pass
        if charges:
            st.session_state.mots_saisis = charges

st.title("🧠 Aide Cémantix Pro — Édition Stratégique")

with st.spinner("Chargement du dictionnaire IA (quelques secondes au démarrage)..."):
    try:
        modele, vecteurs_norm, vecteurs_courants, limite_vocab = charger_donnees()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

if "mots_saisis" not in st.session_state:
    st.session_state.mots_saisis = []
    charger_depuis_url(modele)

# --- BARRE LATÉRALE : PARAMÈTRES & OUTILS ---
with st.sidebar:
    st.header("⚙️ Réglages de l'algorithme")
    
    diversite = st.slider(
        "Diversité des suggestions", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.35, 
        step=0.05,
        help="0 = mots les plus proches possibles. Plus la valeur est haute, plus les pistes explorées sont variées."
    )
    
    exclure_pluriels = st.checkbox(
        "Exclure les pluriels (-s, -x)", 
        value=True,
        help="Cémantix n'utilise généralement que le singulier."
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
    st.caption("💾 Sauvegarde automatique active dans l'URL de la page.")

# --- ZONE DE DÉPART RAPIDE (MOTS SONDES) ---
with st.expander("🚀 Mots sondes pour démarrer une partie (cliquez pour copier)", expanded=not bool(st.session_state.mots_saisis)):
    st.caption("Ces mots couvrent des domaines très éloignés pour quadriller rapidement le dictionnaire :")
    mots_sondes = ["homme", "temps", "objet", "manger", "pays", "eau", "travail", "maison", "musique", "science"]
    cols_sondes = st.columns(5)
    for i, sonde in enumerate(mots_sondes):
        with cols_sondes[i % 5]:
            st.code(sonde, language=None)

# --- ZONE PRINCIPALE : SAISIE DES INDICES ---
col_saisie, col_lot = st.columns(2)

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
                sauvegarder_url()
                st.rerun()
            else:
                st.warning(f"Le mot **'{mot}'** est absent du dictionnaire.")

with col_lot:
    with st.expander("📋 Import rapide (Copier-coller depuis Cémantix)", expanded=True):
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
                sauvegarder_url()
                st.rerun()

# --- AFFICHAGE DES INDICES SAISIS & DÉTECTION D'ANOMALIES ---
if st.session_state.mots_saisis:
    st.divider()
    header_col, btn_col = st.columns([4, 1])
    with header_col:
        st.subheader(f"📌 Vos indices ({len(st.session_state.mots_saisis)})")
    with btn_col:
        if st.button("🗑️ Tout effacer", use_container_width=True):
            st.session_state.mots_saisis = []
            sauvegarder_url()
            st.rerun()

    indices_ordonnes = sorted(st.session_state.mots_saisis, key=lambda x: x["score"], reverse=True)
    
    cols_indices = st.columns(4)
    for idx_item, item in enumerate(indices_ordonnes):
        with cols_indices[idx_item % 4]:
            c_txt, c_del = st.columns([4, 1])
            temp = item['score'] * 100
            emoji = "🔥" if temp >= 40 else ("☀️" if temp >= 20 else ("🌤️" if temp > 0 else "❄️"))
            c_txt.markdown(f"{emoji} **{item['mot']}** : `{temp:.2f} °C`")
            if c_del.button("❌", key=f"del_{item['mot']}", help=f"Supprimer {item['mot']}"):
                st.session_state.mots_saisis = [x for x in st.session_state.mots_saisis if x["mot"] != item["mot"]]
                sauvegarder_url()
                st.rerun()

    # --- CALCUL DES MEILLEURS CANDIDATS ---
    erreur_quadratique = np.zeros(limite_vocab)
    somme_poids = 0.0
    erreurs_par_indice = []

    for item in st.session_state.mots_saisis:
        idx_mot = modele.key_to_index[item["mot"]]
        vec_mot = vecteurs_norm[idx_mot]
        similarites = np.dot(vecteurs_courants, vec_mot)
        
        score_brut = item["score"]
        cible_ajustee = 0.35 + (score_brut - 0.35) * 0.55 if score_brut > 0.35 else score_brut
        poids = (max(0.0, score_brut) + 0.2) ** 3
        
        err_ind = (similarites - cible_ajustee) ** 2
        erreurs_par_indice.append((item["mot"], item["score"] * 100, err_ind))
        
        erreur_quadratique += poids * err_ind
        somme_poids += poids

    indices_tries = np.argsort(erreur_quadratique)
    meilleur_vocab_idx = indices_tries[0]

    # Détecteur de fausses pistes (si au moins 4 indices dont 2 positifs)
    indices_positifs = [e for e in erreurs_par_indice if e[1] > 15.0]
    if len(indices_positifs) >= 3:
        suspects = []
        for m_nom, m_temp, err_vec in indices_positifs:
            if err_vec[meilleur_vocab_idx] > 0.12:
                suspects.append(f"**{m_nom}** ({m_temp:.1f} °C)")
        if suspects:
            st.warning(f"⚠️ **Analyse de cohérence :** {', '.join(suspects)} semble(nt) tirer la recherche dans une direction différente de vos autres mots chauds.")

    st.divider()

    mots_deja_testes = {m["mot"] for m in st.session_state.mots_saisis}
    terminaisons_exclues = ("ait", "aient", "era", "eront", "iez", "ions", "âmes", "âtes", "èrent")
    
    pool_candidats = []
    for idx in indices_tries:
        candidat = modele.index_to_key[idx]
        if (
            candidat not in mots_deja_testes 
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

    resultats = []
    vecteurs_selectionnes = []
    
    while len(resultats) < 30 and pool_candidats:
        meilleur_score_mmr = -999.0
        meilleur_idx_pool = 0
        
        for i, (vocab_idx, mot_cand, corr) in enumerate(pool_candidats):
            if not vecteurs_selectionnes or diversite == 0:
                penalite = 0.0
            else:
                vec_cand = vecteurs_courants[vocab_idx]
                sim_max = max(np.dot(vec_cand, v_sel) for v_sel in vecteurs_selectionnes)
                penalite = diversite * (sim_max * 100)
            
            score_mmr = corr - penalite
            if score_mmr > meilleur_score_mmr:
                meilleur_score_mmr = score_mmr
                meilleur_idx_pool = i
                
        vocab_idx, mot_choisi, corr_choisie = pool_candidats.pop(meilleur_idx_pool)
        resultats.append((vocab_idx, mot_choisi, corr_choisie))
        vecteurs_selectionnes.append(vecteurs_courants[vocab_idx])

    # --- ONGLETS : SUGGESTIONS, BARYCENTRE & CARTE 2D ---
    tab_suggestions, tab_barycentre, tab_carte = st.tabs([
        "🔍 Top 30 des suggestions", 
        "🎯 Centre de gravité (Barycentre)", 
        "🗺️ Carte sémantique 2D"
    ])

    with tab_suggestions:
        st.caption("Cliquez sur l'icône à droite d'un mot pour le copier en un clic :")
        colonnes = st.columns(3)
        for i, (_, mot_candidat, pct) in enumerate(resultats):
            with colonnes[i % 3]:
                st.markdown(f"**#{i + 1}** — Pertinence : `{pct:.1f}%`")
                st.code(mot_candidat, language=None)

    with tab_barycentre:
        st.caption("Additionne purement les vecteurs de vos 5 mots les plus chauds pour trouver le mot situé exactement à leur carrefour.")
        top_chauds = [m for m in indices_ordonnes if m["score"] > 0][:5]
        if len(top_chauds) < 2:
            st.info("Ajoutez au moins 2 indices positifs (> 0 °C) pour calculer leur centre de gravité.")
        else:
            vec_barycentre = np.zeros(vecteurs_norm.shape[1])
            for m in top_chauds:
                poids_b = max(0.05, m["score"]) ** 2
                vec_barycentre += vecteurs_norm[modele.key_to_index[m["mot"]]] * poids_b
            
            norme = np.linalg.norm(vec_barycentre)
            if norme > 0:
                vec_barycentre /= norme
                sims_bary = np.dot(vecteurs_courants, vec_barycentre)
                idx_bary = np.argsort(sims_bary)[::-1]
                
                mots_bary = []
                for idx in idx_bary:
                    cand = modele.index_to_key[idx]
                    if (
                        cand not in mots_deja_testes
                        and cand.isalpha()
                        and cand.islower()
                        and len(cand) >= 3
                        and (not exclure_pluriels or not cand.endswith(("s", "x")))
                    ):
                        mots_bary.append((cand, sims_bary[idx] * 100))
                    if len(mots_bary) >= 12:
                        break
                
                cols_b = st.columns(3)
                for idx_b, (m_b, sim_b) in enumerate(mots_bary):
                    with cols_b[idx_b % 3]:
                        st.markdown(f"**{idx_b + 1}.** Proximité centrale : `{sim_b:.1f}%`")
                        st.code(m_b, language=None)

    with tab_carte:
        if len(st.session_state.mots_saisis) < 3:
            st.info("Ajoutez au moins 3 indices pour générer la carte sémantique 2D.")
        else:
            mots_carte = []
            vecteurs_carte = []
            couleurs_carte = []
            tailles_carte = []
            symboles_carte = []

            for item in st.session_state.mots_saisis:
                temp_c = item['score'] * 100
                mots_carte.append(f"{item['mot']} ({temp_c:.0f}°C)")
                vecteurs_carte.append(vecteurs_norm[modele.key_to_index[item["mot"]]])
                couleurs_carte.append(temp_c)
                tailles_carte.append(max(10, (item["score"] + 1) * 14))
                symboles_carte.append("Indice testé")

            for _, mot_cand, pct in resultats[:10]:
                mots_carte.append(f"💡 {mot_cand}")
                vecteurs_carte.append(vecteurs_norm[modele.key_to_index[mot_cand]])
                couleurs_carte.append(65.0)
                tailles_carte.append(13)
                symboles_carte.append("Suggestion Top 10")

            pca = PCA(n_components=2)
            coords = pca.fit_transform(np.array(vecteurs_carte))
            variance_totale = sum(pca.explained_variance_ratio_) * 100

            fig = px.scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                text=mots_carte,
                color=couleurs_carte,
                symbol=symboles_carte,
                size=tailles_carte,
                color_continuous_scale="RdYlBu_r",
                range_color=[-20, 80],
                labels={
                    "x": "Axe sémantique 1", 
                    "y": "Axe sémantique 2", 
                    "color": "Température (°C)",
                    "symbol": "Type"
                }
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=580)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"ℹ️ **Fiabilité de la projection 2D :** Ce graphique conserve **{variance_totale:.1f} %** de la distance sémantique réelle entre vos mots.")
