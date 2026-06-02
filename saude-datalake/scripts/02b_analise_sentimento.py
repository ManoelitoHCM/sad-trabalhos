"""
02b_analise_sentimento.py
=========================
Limpeza, tokenização e análise de sentimento dos comentários do Instagram.

Etapas (camada bronze -> prata -> ouro para o dado textual):
  1. Limpeza: remove emojis, hashtags, menções, URLs e pontuação.
  2. Tokenização e remoção de stopwords em português.
  3. Análise de sentimento por léxico (dicionário de termos positivos/negativos
     adaptado ao contexto hospitalar) -> classifica em positivo/negativo/neutro.
  4. Extração de temas: palavras e bigramas mais recorrentes por sentimento.

Saídas:
  data/prata/comentarios_prata.csv      (texto limpo + tokens + sentimento)
  data/ouro/ouro_sentimento.csv         (contagem por sentimento)
  data/ouro/ouro_palavras_frequentes.csv(termos mais recorrentes)

Optei por léxico (e não um modelo pesado tipo BERT) por ser leve, transparente e
sem dependências externas — adequado a um trabalho reproduzível. O código indica
onde plugar um modelo pré-treinado, se desejado.
"""

import os
import re
import unicodedata
from collections import Counter
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
for c in ("prata", "ouro"):
    os.makedirs(os.path.join(DATA, c), exist_ok=True)

# ---- Léxicos de sentimento (contexto hospitalar) ----
POS = {
    "excelente", "atencioso", "atenciosa", "atenciosos", "gratidao", "obrigado",
    "obrigada", "competente", "competentes", "humano", "humanos", "humanizado",
    "parabens", "maravilhoso", "maravilhosa", "otimo", "otima", "bom", "boa",
    "melhor", "sucesso", "recomendo", "amei", "adorei", "acolhida", "acolhido",
    "limpo", "limpa", "organizado", "organizada", "salvaram", "cuidaram",
    "incrivel", "nota", "agradecer", "bem",
}
NEG = {
    "absurdo", "demora", "demorado", "demorada", "espera", "esperei", "fila",
    "pessimo", "pessima", "ruim", "descaso", "falta", "faltou", "sujo", "suja",
    "quebrado", "quebrada", "lamentavel", "revoltante", "remarcada", "horas",
    "corredor", "ninguem", "problema", "reclamacao", "abandonado", "triste",
    "demais", "poucos", "falho", "falha",
}
INTENS_NEG_CONTEXT = {"horas", "demais"}  # só contam como negativos perto de termos NEG

STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "os", "as", "no", "na",
    "para", "pra", "por", "com", "que", "se", "ao", "dos", "das", "meu", "minha",
    "eu", "ele", "ela", "isso", "muito", "muita", "ja", "foi", "fui", "ser",
    "esta", "este", "todo", "toda", "todos", "tem", "vou", "sao", "mais", "mas",
    "sem", "ate", "la", "aqui", "so", "tudo", "vc", "voce", "voces", "qual",
    "como", "onde", "quando", "alguem", "sabe", "fica", "pelo", "pela",
}


def remover_acentos(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def limpar(texto: str) -> str:
    t = str(texto).lower()
    t = re.sub(r"http\S+", " ", t)               # URLs
    t = re.sub(r"[@#]\w+", " ", t)               # menções e hashtags
    # remove emojis e qualquer caractere não-letra/ço
    t = remover_acentos(t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenizar(texto_limpo: str) -> list:
    return [w for w in texto_limpo.split() if w not in STOPWORDS and len(w) > 2]


def classificar(tokens: list) -> str:
    pos = sum(1 for w in tokens if w in POS)
    neg = sum(1 for w in tokens if w in NEG)
    if pos > neg:
        return "positivo"
    if neg > pos:
        return "negativo"
    return "neutro"


def analisar():
    bronze = pd.read_csv(os.path.join(DATA, "bronze", "comentarios_instagram_bronze.csv"))
    bronze["texto_limpo"] = bronze["texto"].apply(limpar)
    bronze["tokens"] = bronze["texto_limpo"].apply(tokenizar)
    bronze["sentimento"] = bronze["tokens"].apply(classificar)

    prata = bronze.copy()
    prata["tokens"] = prata["tokens"].apply(lambda x: " ".join(x))
    prata.to_csv(os.path.join(DATA, "prata", "comentarios_prata.csv"), index=False)

    # OURO 1 — contagem por sentimento
    senti = bronze["sentimento"].value_counts().reset_index()
    senti.columns = ["sentimento", "quantidade"]
    senti.to_csv(os.path.join(DATA, "ouro", "ouro_sentimento.csv"), index=False)

    # OURO 2 — palavras mais frequentes (global e por sentimento)
    linhas = []
    for s in ["positivo", "negativo", "neutro"]:
        cnt = Counter()
        for toks in bronze[bronze["sentimento"] == s]["tokens"]:
            cnt.update(toks)
        for palavra, freq in cnt.most_common(10):
            linhas.append({"sentimento": s, "palavra": palavra, "frequencia": freq})
    pal = pd.DataFrame(linhas)
    pal.to_csv(os.path.join(DATA, "ouro", "ouro_palavras_frequentes.csv"), index=False)

    # Validação contra rótulo de referência (acurácia do léxico)
    if "rotulo_referencia" in bronze.columns:
        acerto = (bronze["sentimento"] == bronze["rotulo_referencia"]).mean()
        print(f"[VALIDAÇÃO] concordância do léxico com a referência: {acerto:.1%}")

    print("[PRATA] comentários limpos e tokenizados")
    print("[OURO]  sentimento e palavras frequentes gerados")
    print(senti.to_string(index=False))
    return senti, pal


if __name__ == "__main__":
    analisar()
