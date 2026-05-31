"""
preparar_dados.py
=================
Constrói a base de referência limpa do SAD ENCCEJA a partir dos microdados
oficiais do INEP (ENCCEJA 2024, arquivo NACIONAL REGULAR).

Entrada : microdados_encceja_2024.zip  (ou o CSV REG_NAC já extraído)
Saídas  : encceja_2024_limpo_full.csv    (base completa higienizada)
          encceja_2024_referencia.csv    (amostra estratificada, usada pela GUI)

Etapas: filtro de presença -> notas válidas -> recodificação das variáveis
de perfil -> normalização das escalas ordinais (na verdade feita no K-NN) ->
gravação. Veja o relatório (seção 3) para a justificativa de cada decisão.

Uso:  python preparar_dados.py  [caminho_do_zip_ou_csv]
"""

import sys
import zipfile
import numpy as np
import pandas as pd

ENTRADA = sys.argv[1] if len(sys.argv) > 1 else "microdados_encceja_2024.zip"

# Colunas lidas do arquivo bruto (evita carregar as 117 variáveis)
USECOLS = [
    "TP_CERTIFICACAO", "TP_FAIXA_ETARIA", "TP_SEXO", "SG_UF_PROVA",
    "Q44", "Q50", "Q11",
    "TP_PRESENCA_LC", "TP_PRESENCA_CH", "TP_PRESENCA_MT", "TP_PRESENCA_CN",
    "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_REDACAO",
    "IN_APROVADO_LC", "IN_APROVADO_CH", "IN_APROVADO_MT", "IN_APROVADO_CN",
]
PRESENCA = ["TP_PRESENCA_LC", "TP_PRESENCA_CH", "TP_PRESENCA_MT", "TP_PRESENCA_CN"]
NOTAS4 = ["NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_MT", "NU_NOTA_CN"]
APROV = ["IN_APROVADO_LC", "IN_APROVADO_CH", "IN_APROVADO_MT", "IN_APROVADO_CN"]

MAPA_RENDA = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
MAPA_ESC = {c: i + 1 for i, c in enumerate("ABCDEFGHIJK")}


def abrir_fonte(caminho):
    """Retorna um handle de arquivo para o CSV REG_NAC, dentro ou fora do zip."""
    if caminho.lower().endswith(".zip"):
        z = zipfile.ZipFile(caminho)
        nome = [n for n in z.namelist() if n.endswith("REG_NAC.csv")][0]
        return z.open(nome)
    return open(caminho, "rb")


def main():
    print(f"Lendo: {ENTRADA}")
    partes, total = [], 0
    leitor = pd.read_csv(abrir_fonte(ENTRADA), sep=";", encoding="latin-1",
                         usecols=USECOLS, dtype=str, chunksize=200_000)
    for ch in leitor:
        total += len(ch)
        for c in PRESENCA:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        # 1) Filtro de presença nas 4 áreas objetivas
        ch = ch[(ch[PRESENCA] == 1).all(axis=1)].copy()
        partes.append(ch)
    df = pd.concat(partes, ignore_index=True)
    print(f"  registros no arquivo: {total:,}")
    print(f"  presentes nas 4 áreas: {len(df):,}")

    # Conversões numéricas
    num = NOTAS4 + ["NU_NOTA_REDACAO", "TP_FAIXA_ETARIA", "TP_CERTIFICACAO"] + APROV
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Notas válidas (existentes e > 0)
    df = df[df[NOTAS4].notna().all(axis=1) & (df[NOTAS4] > 0).all(axis=1)]
    # 3) Campos de perfil obrigatórios
    df = df.dropna(subset=["TP_CERTIFICACAO", "TP_FAIXA_ETARIA", "TP_SEXO", "SG_UF_PROVA"])
    # 4) Recodificações
    df = df[df["Q44"].isin(["A", "B", "C"])]
    df["trabalha"] = np.where(df["Q44"].isin(["A", "B"]), "Sim", "Não")
    df = df[df["Q50"].isin(list(MAPA_RENDA) + ["H"])]
    df["renda_ord"] = df["Q50"].map(MAPA_RENDA)
    df["renda_ord"] = df["renda_ord"].fillna(int(df["renda_ord"].median()))  # "não sei"
    df["escolaridade_ord"] = df["Q11"].map(MAPA_ESC).fillna(0).astype(int)
    df["n_areas_aprovadas"] = df[APROV].sum(axis=1).astype(int)
    print(f"  base higienizada: {len(df):,}")

    limpo = pd.DataFrame({
        "certificacao": df["TP_CERTIFICACAO"].astype(int),
        "faixa_etaria": df["TP_FAIXA_ETARIA"].astype(int),
        "sexo": df["TP_SEXO"],
        "uf": df["SG_UF_PROVA"],
        "trabalha": df["trabalha"],
        "renda_ord": df["renda_ord"].astype(int),
        "escolaridade_ord": df["escolaridade_ord"].astype(int),
        "nota_lc": df["NU_NOTA_LC"].round().astype(int),
        "nota_ch": df["NU_NOTA_CH"].round().astype(int),
        "nota_mt": df["NU_NOTA_MT"].round().astype(int),
        "nota_cn": df["NU_NOTA_CN"].round().astype(int),
        "nota_redacao": df["NU_NOTA_REDACAO"],
        "aprovado_lc": df["IN_APROVADO_LC"].astype(int),
        "aprovado_ch": df["IN_APROVADO_CH"].astype(int),
        "aprovado_mt": df["IN_APROVADO_MT"].astype(int),
        "aprovado_cn": df["IN_APROVADO_CN"].astype(int),
        "n_areas_aprovadas": df["n_areas_aprovadas"].astype(int),
    })
    limpo.to_csv("encceja_2024_limpo_full.csv", index=False)

    # Amostra estratificada por nível de certificação (repo leve + GUI rápida)
    frac = min(1.0, 45_000 / len(limpo))
    amostra = (limpo.groupby("certificacao", group_keys=False)[limpo.columns]
               .apply(lambda g: g.sample(frac=frac, random_state=42))
               .reset_index(drop=True))
    amostra.to_csv("encceja_2024_referencia.csv", index=False)
    print(f"  base completa salva: encceja_2024_limpo_full.csv ({len(limpo):,})")
    print(f"  amostra salva:       encceja_2024_referencia.csv ({len(amostra):,})")


if __name__ == "__main__":
    main()
