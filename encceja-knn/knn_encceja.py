"""
knn_encceja.py
==============
Núcleo do Sistema de Apoio à Decisão (SAD) baseado em K-Nearest Neighbors (K-NN)
para o ENCCEJA 2024.

A ideia central:
    Dado o PERFIL SOCIOECONÔMICO de um novo candidato do cursinho (para o qual NÃO
    conhecemos o desempenho), encontramos os k participantes mais semelhantes do
    histórico do ENCCEJA e usamos o desempenho deles para:
        (1) estimar as notas esperadas do candidato (regressão = média dos vizinhos);
        (2) estimar o risco de reprovação por área (classificação = proporção de
            vizinhos aprovados);
        (3) embasar recomendações pedagógicas ao gestor.

O algoritmo K-NN é implementado "na mão" com NumPy (sem caixa-preta) para deixar
explícito o cálculo de distância, a normalização e a seleção dos vizinhos.

Autor: Manoelito  |  Disciplina: Sistemas de Apoio à Tomada de Decisão
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Metadados de domínio (rótulos legíveis para a interface e os relatórios)
# --------------------------------------------------------------------------- #
AREAS = ["lc", "ch", "mt", "cn"]
AREAS_NOME = {
    "lc": "Linguagens e Códigos",
    "ch": "Ciências Humanas",
    "mt": "Matemática",
    "cn": "Ciências da Natureza",
}

# UF -> Região (reduz a dimensionalidade de 27 UFs para 5 regiões no cálculo
# de distância, evitando o "efeito de esparsidade" do one-hot por estado).
UF_REGIAO = {
    "AC": "N", "AP": "N", "AM": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE",
    "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}
REGIOES = ["N", "NE", "CO", "SE", "S"]

# Rótulos das variáveis ordinais usadas na entrada da interface
RENDA_LABELS = {
    0: "Nenhuma renda", 1: "Até 1 salário mínimo", 2: "De 1 a 2 SM",
    3: "De 2 a 3 SM", 4: "De 3 a 4 SM", 5: "De 4 a 5 SM", 6: "Acima de 5 SM",
}
ESCOLARIDADE_LABELS = {
    0: "Nunca frequentou / não informado",
    1: "1ª série EF", 2: "2ª série EF", 3: "3ª série EF", 4: "4ª série EF",
    5: "5ª série EF", 6: "6ª série EF", 7: "7ª série EF", 8: "8ª série EF",
    9: "1ª série EM", 10: "2ª série EM", 11: "3ª série EM",
}
FAIXA_ETARIA_LABELS = {
    1: "Menor de 17", 2: "17 anos", 3: "18 anos", 4: "19 anos", 5: "20 anos",
    6: "21 anos", 7: "22 anos", 8: "23 anos", 9: "24 anos", 10: "25 anos",
    11: "26 a 30", 12: "31 a 35", 13: "36 a 40", 14: "41 a 45", 15: "46 a 50",
    16: "51 a 55", 17: "56 a 60", 18: "61 a 65", 19: "66 a 70", 20: "Maior de 70",
}

# Peso aplicado às colunas de região (one-hot). Com 1/sqrt(2), a distância
# máxima entre dois perfis de regiões diferentes vale 1.0, mesma escala das
# variáveis ordinais normalizadas em [0, 1]. Evita que a região domine.
PESO_REGIAO = 1.0 / np.sqrt(2)


class ModeloKNN:
    """K-NN para apoio à decisão sobre candidatos do ENCCEJA.

    O modelo NÃO é treinado no sentido tradicional: ele apenas memoriza a base
    de referência (lazy learning) e guarda os parâmetros de normalização
    (mín./máx. das variáveis ordinais) calculados sobre essa base.
    """

    def __init__(self, base_csv: str, k: int = 15):
        self.k = k
        self.df = pd.read_csv(base_csv)
        # Garante região disponível para filtragem/exibição
        self.df["regiao"] = self.df["uf"].map(UF_REGIAO)
        # Parâmetros de normalização (min-max) das ordinais — fixos a partir da base
        self._minmax = {
            "faixa_etaria": (1, 20),
            "renda_ord": (0, 6),
            "escolaridade_ord": (0, 11),
        }
        # Pré-computa as matrizes de features por nível de certificação,
        # pois um candidato só é comparado a participantes do MESMO nível
        # (as provas de Fundamental e Médio são diferentes).
        self._cache = {}
        for cert in (1, 2):
            sub = self.df[self.df["certificacao"] == cert].reset_index(drop=True)
            if len(sub):
                self._cache[cert] = (sub, self._matriz_features(sub))

    # ----------------------------------------------------------------- #
    # Construção/normalização do vetor de atributos
    # ----------------------------------------------------------------- #
    def _norm(self, serie, var):
        lo, hi = self._minmax[var]
        return (serie - lo) / (hi - lo)

    def _matriz_features(self, d: pd.DataFrame) -> np.ndarray:
        """Monta a matriz numérica de atributos já normalizada em [0, 1]."""
        cols = [
            self._norm(d["faixa_etaria"], "faixa_etaria"),
            self._norm(d["renda_ord"], "renda_ord"),
            self._norm(d["escolaridade_ord"], "escolaridade_ord"),
            (d["sexo"] == "M").astype(float),       # binária
            (d["trabalha"] == "Sim").astype(float),  # binária
        ]
        for r in REGIOES:                            # one-hot de região ponderado
            cols.append((d["regiao"] == r).astype(float) * PESO_REGIAO)
        return np.column_stack(cols).astype(float)

    def _vetor_candidato(self, perfil: dict) -> np.ndarray:
        """Transforma o dicionário de entrada da interface em um vetor de features."""
        reg = UF_REGIAO.get(perfil["uf"])
        base = [
            (perfil["faixa_etaria"] - 1) / 19,
            perfil["renda_ord"] / 6,
            perfil["escolaridade_ord"] / 11,
            1.0 if perfil["sexo"] == "M" else 0.0,
            1.0 if perfil["trabalha"] == "Sim" else 0.0,
        ]
        for r in REGIOES:
            base.append((1.0 if reg == r else 0.0) * PESO_REGIAO)
        return np.array(base, dtype=float)

    # ----------------------------------------------------------------- #
    # Predição (o coração do K-NN)
    # ----------------------------------------------------------------- #
    def prever(self, perfil: dict, k: int | None = None) -> dict:
        """Retorna notas esperadas, risco de aprovação e dados dos vizinhos."""
        k = k or self.k
        cert = perfil["certificacao"]
        if cert not in self._cache:
            raise ValueError("Sem participantes de referência para este nível.")
        sub, X = self._cache[cert]
        v = self._vetor_candidato(perfil)

        # Distância euclidiana do candidato a TODOS os participantes do mesmo nível
        dist = np.sqrt(((X - v) ** 2).sum(axis=1))
        # k menores distâncias (vizinhos mais próximos)
        k = min(k, len(sub))
        idx = np.argpartition(dist, k - 1)[:k]
        idx = idx[np.argsort(dist[idx])]            # ordena do mais próximo
        viz = sub.iloc[idx].copy()
        viz["distancia"] = dist[idx]

        # Estimativas por área: nota esperada (regressão) e risco (classificação)
        resultado = {"k": k, "vizinhos": viz, "areas": {}}
        for a in AREAS:
            notas = viz[f"nota_{a}"].values
            aprov = viz[f"aprovado_{a}"].values
            media_pop = sub[f"nota_{a}"].mean()      # média do nível, p/ comparação
            resultado["areas"][a] = {
                "nota_esperada": float(notas.mean()),
                "nota_min_viz": float(notas.min()),
                "nota_max_viz": float(notas.max()),
                "taxa_aprovacao_viz": float(aprov.mean()),
                "media_populacao": float(media_pop),
                "vs_media": float(notas.mean() - media_pop),
            }
        # Redação (escala 0–10) e visão agregada
        resultado["nota_redacao_esperada"] = float(viz["nota_redacao"].mean())
        taxas = [resultado["areas"][a]["taxa_aprovacao_viz"] for a in AREAS]
        resultado["taxa_aprovacao_media"] = float(np.mean(taxas))
        resultado["n_areas_esperadas"] = float(viz["n_areas_aprovadas"].mean())
        resultado["nivel_risco"] = self._classificar_risco(np.mean(taxas))
        return resultado

    @staticmethod
    def _classificar_risco(taxa_media: float) -> str:
        if taxa_media >= 0.75:
            return "BAIXO"
        if taxa_media >= 0.50:
            return "MÉDIO"
        return "ALTO"


def gerar_recomendacoes(resultado: dict) -> list[str]:
    """Traduz a saída numérica do K-NN em recomendações gerenciais (texto)."""
    recs = []
    risco = resultado["nivel_risco"]
    taxa = resultado["taxa_aprovacao_media"]

    if risco == "ALTO":
        recs.append(
            "Perfil de ALTO RISCO de reprovação: a maioria dos candidatos "
            "semelhantes não foi aprovada. Recomenda-se reforço intensivo e "
            "acompanhamento pedagógico individual desde a matrícula."
        )
    elif risco == "MÉDIO":
        recs.append(
            "Perfil de RISCO MÉDIO: resultado dos semelhantes é dividido. "
            "Recomenda-se acompanhamento regular e reforço direcionado às áreas "
            "mais fracas listadas abaixo."
        )
    else:
        recs.append(
            "Perfil de BAIXO RISCO: a maioria dos candidatos semelhantes foi "
            "aprovada. O foco pode ser manter o ritmo e monitorar pontualmente "
            "as áreas mais frágeis."
        )

    # Ordena áreas da mais fraca para a mais forte (por taxa de aprovação)
    areas_ord = sorted(
        resultado["areas"].items(), key=lambda kv: kv[1]["taxa_aprovacao_viz"]
    )
    fracas = [AREAS_NOME[a] for a, d in areas_ord if d["taxa_aprovacao_viz"] < 0.6]
    if fracas:
        recs.append(
            "Priorizar reforço em: " + ", ".join(fracas) +
            " (áreas em que os candidatos semelhantes mais reprovaram)."
        )

    pior = areas_ord[0]
    recs.append(
        f"Disciplina de maior atenção: {AREAS_NOME[pior[0]]} "
        f"(nota esperada {pior[1]['nota_esperada']:.0f}, "
        f"{pior[1]['taxa_aprovacao_viz']*100:.0f}% dos vizinhos aprovados)."
    )

    if resultado["nota_redacao_esperada"] < 5.0:
        recs.append(
            f"Redação esperada baixa ({resultado['nota_redacao_esperada']:.1f}/10): "
            "incluir produção textual orientada no plano de estudo."
        )

    recs.append(
        f"Expectativa global: aprovação em ~{resultado['n_areas_esperadas']:.1f} "
        f"de 4 áreas (taxa média de aprovação dos vizinhos: {taxa*100:.0f}%)."
    )
    return recs


if __name__ == "__main__":
    # Teste rápido via linha de comando
    m = ModeloKNN("encceja_2024_referencia.csv", k=15)
    perfil = dict(certificacao=2, faixa_etaria=12, sexo="F", uf="CE",
                  trabalha="Sim", renda_ord=1, escolaridade_ord=9)
    r = m.prever(perfil)
    print("Nível de risco:", r["nivel_risco"],
          "| taxa média:", round(r["taxa_aprovacao_media"], 3))
    for a in AREAS:
        d = r["areas"][a]
        print(f"  {AREAS_NOME[a]:22} nota~{d['nota_esperada']:5.0f} "
              f"(vs média {d['vs_media']:+.0f}) | aprov vizinhos {d['taxa_aprovacao_viz']*100:.0f}%")
    print("\nRecomendações:")
    for rec in gerar_recomendacoes(r):
        print(" -", rec)
