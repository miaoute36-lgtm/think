"""
avis_3_ia.py
Application Streamlit – Avis croisés de 3 IA + synthèse finale

  Gemini  → avis + lecture native des PDF/images   (Google Gemini 3.6 Flash — gratuit)
  Groq    → avis avec raisonnement approfondi       (GPT-OSS 120B — gratuit)
  Mistral → avis + synthèse finale dédiée           (Mistral Small — gratuit)

Chaque avis est structuré : texte, note /10, avantages, inconvénients.
Formats acceptés en pièce jointe : PDF, DOCX, JPG, PNG, TXT, CSV, XLSX

Lancement : streamlit run avis_3_ia.py
"""

import streamlit as st
import requests
import os
import io
import json
import re
import base64
import concurrent.futures

import pandas as pd
from pypdf import PdfReader
from docx import Document

# ─── Modèles & URLs ────────────────────────────────────────────────────────
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

MODEL_GEMINI  = "gemini-3.6-flash"
MODEL_GROQ    = "openai/gpt-oss-120b"
MODEL_MISTRAL = "mistral-small-latest"

MAX_TOKENS       = 900
MAX_TOKENS_GEMINI = 4096  # le "thinking" de Gemini 3.x peut consommer une grande partie du budget de façon imprévisible
MAX_TOKENS_SYN   = 1100
CONTEXTE_MAX    = 6000   # chars max du contenu texte des fichiers injecté

# ─── Consigne commune de format (JSON structuré) ───────────────────────────
CONSIGNE_FORMAT = (
    "Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, "
    "sans balises markdown ```, exactement au format suivant :\n"
    '{"avis": "ton avis développé en 3-6 phrases", "note": 7, '
    '"avantages": ["point fort 1", "point fort 2", "point fort 3"], '
    '"inconvenients": ["point faible 1", "point faible 2"]}\n'
    "La note est un entier entre 0 et 10. Réponds toujours en français."
)

PROMPT_GEMINI = (
    "Tu donnes un avis argumenté, factuel et nuancé sur le sujet/produit/question soumis. "
    "Si des documents ou images sont fournis, appuie-toi dessus explicitement. " + CONSIGNE_FORMAT
)
PROMPT_GROQ = (
    "Tu donnes un avis fondé sur un raisonnement structuré et approfondi : "
    "identifie les enjeux, les implications à moyen/long terme, les angles morts. " + CONSIGNE_FORMAT
)
PROMPT_MISTRAL = (
    "Tu donnes un avis critique et pragmatique, orienté décision concrète : "
    "serait-ce un bon choix, dans quel contexte, pour qui. " + CONSIGNE_FORMAT
)

PROMPT_SYNTHESE = (
    "Tu reçois trois avis indépendants (JSON) donnés par trois IA différentes sur le même "
    "sujet/produit/question. Rédige une synthèse consolidée. "
    "Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, "
    "sans balises markdown, exactement au format suivant :\n"
    '{"avis_global": "synthèse en 4-8 phrases qui tranche et conseille concrètement", '
    '"note_globale": 7, '
    '"convergences": ["point sur lequel les 3 avis s\'accordent"], '
    '"divergences": ["point sur lequel les avis diffèrent, avec explication"]}\n'
    "La note est un entier entre 0 et 10. Réponds toujours en français."
)

# ─── Récupération des clés API (secrets Streamlit > variables d'environnement) ─
def get_secret(key: str) -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


# ─── Init session ───────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "gemini_key":       get_secret("GEMINI_API_KEY"),
        "groq_key":         get_secret("GROQ_API_KEY"),
        "mistral_key":      get_secret("MISTRAL_API_KEY"),
        "fichiers_contexte": "",
        "fichiers_gemini":   [],   # [(mime, base64)] — PDF/images envoyés natifs à Gemini
        "sujet":            "",
        "res_gemini":   None,
        "res_groq":     None,
        "res_mistral":  None,
        "res_synthese": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Extraction de texte (pour Groq & Mistral, qui ne lisent pas les fichiers) ─
def extraire_texte(fichier) -> str:
    nom = fichier.name.lower()
    try:
        if nom.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(fichier.read()))
            pages = [p.extract_text() or "" for p in reader.pages]
            texte = "\n".join(pages).strip()
            return texte[:4000] if texte else "[PDF sans texte extractible]"
        elif nom.endswith(".docx"):
            doc = Document(io.BytesIO(fichier.read()))
            texte = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return texte[:4000]
        elif nom.endswith((".xlsx", ".xlsm")):
            xl = pd.ExcelFile(io.BytesIO(fichier.read()), engine="openpyxl")
            parties = [f"[Feuille : {s}]\n{xl.parse(s).to_string(index=False)}" for s in xl.sheet_names]
            return "\n\n".join(parties)[:4000]
        elif nom.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(fichier.read()))
            return df.to_string(index=False)[:4000]
        elif nom.endswith(".txt"):
            return fichier.read().decode("utf-8", errors="ignore")[:4000]
        else:
            return ""
    except Exception as e:
        return f"[Erreur lecture {fichier.name} : {e}]"


def est_pdf_ou_image(nom: str) -> bool:
    return nom.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))


def mime_de(nom: str) -> str:
    nom = nom.lower()
    if nom.endswith(".pdf"):
        return "application/pdf"
    if nom.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if nom.endswith(".png"):
        return "image/png"
    return "application/octet-stream"


# ─── Extraction JSON robuste (au cas où le modèle ajoute du texte autour) ──
def parser_json_reponse(texte: str) -> dict:
    if not texte:
        return {}
    # Retire d'éventuelles balises markdown ```json ... ```
    nettoye = re.sub(r"```(json)?", "", texte).strip()
    try:
        return json.loads(nettoye)
    except Exception:
        pass
    # Recherche du premier bloc { ... } complet
    match = re.search(r"\{.*\}", nettoye, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Dernier recours : tente de récupérer les champs individuellement,
    # même si le JSON est incomplet (ex. réponse tronquée par la limite de tokens)
    avis_m = re.search(r'"avis"\s*:\s*"((?:[^"\\]|\\.)*)"', nettoye)
    if not avis_m:
        # Le champ "avis" est peut-être tronqué avant même sa guillemet fermante
        avis_m = re.search(r'"avis"\s*:\s*"((?:[^"\\]|\\.)*)$', nettoye)
    note_m = re.search(r'"note"\s*:\s*(\d+)', nettoye)
    avantages_m = re.findall(r'"avantages"\s*:\s*\[(.*?)\]', nettoye, re.DOTALL)
    inconv_m = re.findall(r'"inconvenients"\s*:\s*\[(.*?)\]', nettoye, re.DOTALL)

    def _extraire_liste(bloc):
        if not bloc:
            return []
        return [x.strip() for x in re.findall(r'"((?:[^"\\]|\\.)*)"', bloc[0])]

    if avis_m or note_m:
        return {
            "avis": (avis_m.group(1).replace('\\"', '"') if avis_m else texte.strip()),
            "note": int(note_m.group(1)) if note_m else None,
            "avantages": _extraire_liste(avantages_m),
            "inconvenients": _extraire_liste(inconv_m),
        }

    # Échec total : on renvoie le texte brut comme avis, sans note ni listes
    return {"avis": texte.strip(), "note": None, "avantages": [], "inconvenients": []}


# ─── Appels API ──────────────────────────────────────────────────────────────
def appel_gemini(question_complete, fichiers_gemini, api_key, system_prompt=PROMPT_GEMINI, max_tok=MAX_TOKENS_GEMINI):
    url = GEMINI_URL.format(model=MODEL_GEMINI)
    parts = [{"text": system_prompt + "\n\n--- DEMANDE ---\n" + question_complete}]
    for mime, b64 in fichiers_gemini:
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_tok,
            "temperature": 0.7,
            "thinkingConfig": {"thinkingLevel": "low"},  # limite la réflexion interne pour laisser de la place à la réponse
        },
    }
    # Clé transmise en en-tête plutôt qu'en paramètre d'URL : elle n'apparaît
    # ainsi jamais dans une éventuelle URL affichée dans un message d'erreur.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    derniere_erreur = None
    for tentative, budget in enumerate([max_tok, max_tok * 2]):  # 2e essai avec budget doublé si troncature
        payload["generationConfig"]["maxOutputTokens"] = budget
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 503 and tentative == 0:
                derniere_erreur = "Service Gemini temporairement surchargé"
                continue
            r.raise_for_status()
            data = r.json()
            candidat = data["candidates"][0]
            texte = candidat["content"]["parts"][0]["text"] if candidat.get("content", {}).get("parts") else ""
            # Si la réponse a été coupée par le budget de tokens (réflexion interne trop gourmande)
            # et qu'il reste une tentative, on relance avec plus de marge plutôt que de renvoyer un texte tronqué.
            if candidat.get("finishReason") == "MAX_TOKENS" and tentative == 0:
                derniere_erreur = "Réponse tronquée (réflexion interne trop longue)"
                continue
            return texte
        except requests.exceptions.HTTPError:
            raise RuntimeError(f"Gemini a répondu avec le code {r.status_code} (service indisponible ou clé invalide). Réessaie dans quelques instants.")
        except requests.exceptions.RequestException:
            raise RuntimeError("Impossible de joindre le service Gemini (connexion). Réessaie dans quelques instants.")
    raise RuntimeError(f"{derniere_erreur} — réessaie, ou reformule la demande plus simplement.")


def appel_groq(question_complete, api_key, system_prompt=PROMPT_GROQ, max_tok=MAX_TOKENS):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_GROQ,
        "max_tokens": max_tok,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question_complete},
        ],
    }
    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def appel_mistral(question_complete, api_key, system_prompt=PROMPT_MISTRAL, max_tok=MAX_TOKENS):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_MISTRAL,
        "max_tokens": max_tok,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question_complete},
        ],
    }
    r = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ─── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Avis croisés de 3 IA", page_icon="🎯", layout="wide")

st.title("🎯 Avis croisés de 3 IA")
st.caption("Gemini 3.6 Flash · Groq (GPT-OSS 120B) · Mistral Small — avis indépendants + synthèse finale")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔑 Clés API")

    cles_preconfigurees = bool(
        get_secret("GEMINI_API_KEY") and get_secret("GROQ_API_KEY") and get_secret("MISTRAL_API_KEY")
    )

    if cles_preconfigurees:
        st.success("🔒 Clés chargées automatiquement (secrets)")
        with st.expander("Modifier les clés pour cette session"):
            gemini_input  = st.text_input("Clé Gemini",  value=st.session_state["gemini_key"],  type="password", key="in_gemini")
            groq_input    = st.text_input("Clé Groq",    value=st.session_state["groq_key"],    type="password", key="in_groq")
            mistral_input = st.text_input("Clé Mistral", value=st.session_state["mistral_key"], type="password", key="in_mistral")
            if st.button("💾 Utiliser ces clés pour la session", use_container_width=True):
                st.session_state["gemini_key"]  = gemini_input
                st.session_state["groq_key"]    = groq_input
                st.session_state["mistral_key"] = mistral_input
                st.rerun()
    else:
        st.warning("⚠️ Clés non pré-configurées.")
        gemini_input  = st.text_input("Clé Gemini",  value=st.session_state["gemini_key"],  type="password", key="in_gemini")
        st.caption("👉 [Obtenir clé Gemini (gratuit)](https://aistudio.google.com/apikey)")
        groq_input    = st.text_input("Clé Groq",    value=st.session_state["groq_key"],    type="password", key="in_groq")
        st.caption("👉 [Obtenir clé Groq (gratuit)](https://console.groq.com/keys)")
        mistral_input = st.text_input("Clé Mistral", value=st.session_state["mistral_key"], type="password", key="in_mistral")
        st.caption("👉 [Obtenir clé Mistral (gratuit)](https://console.mistral.ai/api-keys)")

        if st.button("💾 Sauvegarder les clés (session)", use_container_width=True):
            st.session_state["gemini_key"]  = gemini_input
            st.session_state["groq_key"]    = groq_input
            st.session_state["mistral_key"] = mistral_input
            st.rerun()

    st.divider()
    st.info(
        "**Rôles**\n\n"
        "🔵 **Gemini 3.6 Flash** → Avis + lecture native des PDF/images\n\n"
        "🟠 **GPT-OSS 120B (Groq)** → Avis avec raisonnement approfondi\n\n"
        "🟣 **Mistral Small** → Avis + synthèse finale dédiée"
    )
    st.divider()
    st.success("✅ 100% gratuit\nGemini · Groq · Mistral")


# ─── Zone principale ────────────────────────────────────────────────────────
if st.session_state["res_synthese"] is not None:
    if st.button("🔄 Nouvelle question", use_container_width=False):
        for k in ["res_gemini", "res_groq", "res_mistral", "res_synthese", "sujet", "fichiers_contexte"]:
            st.session_state[k] = None if k.startswith("res") else ""
        st.session_state["fichiers_gemini"] = []
        st.rerun()

sujet = st.text_area(
    "💬 Sujet, question ou produit sur lequel tu veux l'avis des 3 IA",
    value=st.session_state["sujet"],
    placeholder="Ex : Que penses-tu de l'aspirateur Dyson V15 ? / Faut-il investir dans le Bitcoin en 2026 ? / Cette clause de contrat est-elle équilibrée ?",
    height=110,
    key="input_sujet",
)

st.markdown("##### 📎 Documents à l'appui *(optionnel)*")
fichiers = st.file_uploader(
    "PDF, Word, images (JPG/PNG), TXT, CSV, XLSX",
    type=["pdf", "docx", "jpg", "jpeg", "png", "txt", "csv", "xlsx", "xlsm"],
    accept_multiple_files=True,
    help="Les PDF et images sont lus nativement par Gemini. Le texte extrait est partagé avec les 3 IA.",
)

if fichiers:
    textes = []
    fichiers_gemini = []
    with st.spinner("Lecture des fichiers…"):
        for f in fichiers:
            if est_pdf_ou_image(f.name):
                contenu = f.read()
                b64 = base64.b64encode(contenu).decode("utf-8")
                fichiers_gemini.append((mime_de(f.name), b64))
                f.seek(0)
                # Pour Groq/Mistral, on tente aussi une extraction texte si c'est un PDF
                if f.name.lower().endswith(".pdf"):
                    texte = extraire_texte(f)
                    if texte and not texte.startswith("[PDF sans texte"):
                        textes.append(f"=== {f.name} ===\n{texte}")
            else:
                texte = extraire_texte(f)
                if texte:
                    textes.append(f"=== {f.name} ===\n{texte}")

    contexte_brut = "\n\n".join(textes)
    st.session_state["fichiers_contexte"] = contexte_brut[:CONTEXTE_MAX] + ("…[tronqué]" if len(contexte_brut) > CONTEXTE_MAX else "")
    st.session_state["fichiers_gemini"] = fichiers_gemini

    nb_total = len(fichiers)
    nb_img_pdf = len(fichiers_gemini)
    st.success(f"✅ {nb_total} fichier(s) chargé(s)" + (f" — {nb_img_pdf} lu(s) nativement par Gemini" if nb_img_pdf else ""))
else:
    st.session_state["fichiers_contexte"] = ""
    st.session_state["fichiers_gemini"] = []

lancer = st.button("🚀 Obtenir les 3 avis", type="primary", use_container_width=True)


# ─── Affichage d'une carte d'avis structurée ───────────────────────────────
def afficher_carte_avis(titre_emoji, titre, avis_dict, erreur=None):
    with st.container(border=True):
        st.markdown(f"#### {titre_emoji} {titre}")
        if erreur:
            st.error(f"Erreur : {erreur}")
            return
        note = avis_dict.get("note")
        if isinstance(note, (int, float)):
            col_note, col_barre = st.columns([1, 3])
            with col_note:
                st.metric("Note", f"{note}/10")
            with col_barre:
                st.progress(min(max(note, 0), 10) / 10)
        avis_texte = avis_dict.get("avis", "").strip()
        if avis_texte:
            st.markdown(avis_texte)
        avantages = avis_dict.get("avantages") or []
        inconvenients = avis_dict.get("inconvenients") or []
        if avantages or inconvenients:
            col_a, col_i = st.columns(2)
            with col_a:
                st.markdown("**✅ Avantages**")
                for a in avantages:
                    st.markdown(f"- {a}")
            with col_i:
                st.markdown("**⚠️ Inconvénients**")
                for i in inconvenients:
                    st.markdown(f"- {i}")


# ─── Traitement ───────────────────────────────────────────────────────────────
if lancer:
    st.session_state["sujet"] = sujet

    if not sujet.strip():
        st.warning("Décris le sujet, la question ou le produit avant de lancer.")
        st.stop()
    manquantes = []
    if not st.session_state["gemini_key"]:  manquantes.append("Gemini")
    if not st.session_state["groq_key"]:    manquantes.append("Groq")
    if not st.session_state["mistral_key"]: manquantes.append("Mistral")
    if manquantes:
        st.error(f"Clé(s) manquante(s) : {', '.join(manquantes)} — saisis-les dans la sidebar et clique 💾 Sauvegarder.")
        st.stop()

    fichiers_contexte = st.session_state.get("fichiers_contexte", "")
    fichiers_gemini    = st.session_state.get("fichiers_gemini", [])

    question_texte = sujet
    if fichiers_contexte:
        question_texte += "\n\n--- CONTENU DES DOCUMENTS FOURNIS ---\n" + fichiers_contexte

    st.divider()

    key_gemini  = st.session_state["gemini_key"]
    key_groq    = st.session_state["groq_key"]
    key_mistral = st.session_state["mistral_key"]

    placeholder_gemini  = st.empty()
    placeholder_groq    = st.empty()
    placeholder_mistral = st.empty()
    placeholder_gemini.info("🔵 Gemini 3.6 Flash — en cours…")
    placeholder_groq.info("🟠 GPT-OSS 120B (Groq) — en cours…")
    placeholder_mistral.info("🟣 Mistral Small — en cours…")

    resultats = {}
    erreurs = {}

    def run_gemini():
        return appel_gemini(question_texte, fichiers_gemini, api_key=key_gemini)

    def run_groq():
        return appel_groq(question_texte, api_key=key_groq)

    def run_mistral():
        return appel_mistral(question_texte, api_key=key_mistral)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_gemini):  "gemini",
            executor.submit(run_groq):    "groq",
            executor.submit(run_mistral): "mistral",
        }
        for future in concurrent.futures.as_completed(futures):
            nom = futures[future]
            try:
                resultats[nom] = future.result()
            except Exception as e:
                erreurs[nom] = str(e)
                resultats[nom] = None

    avis_gemini  = parser_json_reponse(resultats.get("gemini"))  if resultats.get("gemini")  else {}
    avis_groq    = parser_json_reponse(resultats.get("groq"))    if resultats.get("groq")    else {}
    avis_mistral = parser_json_reponse(resultats.get("mistral")) if resultats.get("mistral") else {}

    st.session_state["res_gemini"]  = avis_gemini
    st.session_state["res_groq"]    = avis_groq
    st.session_state["res_mistral"] = avis_mistral

    placeholder_gemini.empty()
    placeholder_groq.empty()
    placeholder_mistral.empty()

    col1, col2, col3 = st.columns(3)
    with col1:
        afficher_carte_avis("🔵", "Gemini 3.6 Flash", avis_gemini, erreurs.get("gemini"))
    with col2:
        afficher_carte_avis("🟠", "GPT-OSS 120B (Groq)", avis_groq, erreurs.get("groq"))
    with col3:
        afficher_carte_avis("🟣", "Mistral Small", avis_mistral, erreurs.get("mistral"))

    # ── Synthèse finale ───────────────────────────────────────────────────────
    if avis_gemini or avis_groq or avis_mistral:
        st.divider()
        st.markdown("### 🟢 Synthèse finale — Mistral Small (appel dédié)")

        prompt_synthese_user = (
            f"SUJET : {sujet}\n\n"
            f"AVIS GEMINI 3.6 FLASH :\n{json.dumps(avis_gemini, ensure_ascii=False)}\n\n"
            f"AVIS GPT-OSS 120B (GROQ) :\n{json.dumps(avis_groq, ensure_ascii=False)}\n\n"
            f"AVIS MISTRAL SMALL :\n{json.dumps(avis_mistral, ensure_ascii=False)}"
        )

        with st.spinner("🟢 Synthèse en cours…"):
            try:
                synthese_brute = appel_mistral(
                    prompt_synthese_user, api_key=key_mistral,
                    system_prompt=PROMPT_SYNTHESE, max_tok=MAX_TOKENS_SYN,
                )
                synthese = parser_json_reponse(synthese_brute)
                st.session_state["res_synthese"] = synthese

                notes = [d.get("note") for d in [avis_gemini, avis_groq, avis_mistral]
                         if isinstance(d.get("note"), (int, float))]
                note_moyenne = round(sum(notes) / len(notes), 1) if notes else None

                with st.container(border=True):
                    col_note1, col_note2 = st.columns(2)
                    if isinstance(synthese.get("note_globale"), (int, float)):
                        col_note1.metric("Note de synthèse (IA)", f"{synthese['note_globale']}/10")
                    if note_moyenne is not None:
                        col_note2.metric("Moyenne simple des 3 notes", f"{note_moyenne}/10")

                    if synthese.get("avis_global"):
                        st.markdown(synthese["avis_global"])

                    convergences = synthese.get("convergences") or []
                    divergences  = synthese.get("divergences") or []
                    if convergences:
                        st.markdown("**🤝 Points de convergence**")
                        for c in convergences:
                            st.markdown(f"- {c}")
                    if divergences:
                        st.markdown("**⚖️ Points de divergence**")
                        for d in divergences:
                            st.markdown(f"- {d}")
            except Exception as e:
                st.error(f"Erreur synthèse : {e}")

    st.divider()
    if st.button("🔄 Nouvelle question", use_container_width=True, key="btn_bas"):
        for k in ["res_gemini", "res_groq", "res_mistral", "res_synthese", "sujet", "fichiers_contexte"]:
            st.session_state[k] = None if k.startswith("res") else ""
        st.session_state["fichiers_gemini"] = []
        st.rerun()


# ─── Affichage persistant après rerun ───────────────────────────────────────
elif st.session_state["res_synthese"] is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        afficher_carte_avis("🔵", "Gemini 3.6 Flash", st.session_state["res_gemini"] or {})
    with col2:
        afficher_carte_avis("🟠", "GPT-OSS 120B (Groq)", st.session_state["res_groq"] or {})
    with col3:
        afficher_carte_avis("🟣", "Mistral Small", st.session_state["res_mistral"] or {})

    st.divider()
    st.markdown("### 🟢 Synthèse finale")
    synthese = st.session_state["res_synthese"] or {}
    with st.container(border=True):
        if isinstance(synthese.get("note_globale"), (int, float)):
            st.metric("Note de synthèse", f"{synthese['note_globale']}/10")
        if synthese.get("avis_global"):
            st.markdown(synthese["avis_global"])


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Application locale – aucune donnée stockée sur disque. "
    "Les clés API et les documents transitent uniquement vers les serveurs Gemini, Groq et Mistral."
)
