"""
01_etl_internacoes.py
=====================
Pipeline de dados estruturados do Data Lake — Hospital HUOL/UFRN.

Implementa as três camadas do Data Lake para os dados de internação do SUS:

  BRONZE : ingestão crua dos 8 CSVs trimestrais (Jan/2024–Dez/2025), unidos
           sem nenhuma transformação, preservando o dado de origem.
  PRATA  : dados limpos e padronizados — acentuação corrigida, nomes de
           especialidades unificados, faixas etárias criadas, sexo normalizado,
           datas convertidas e registros inválidos tratados.
  OURO   : tabelas analíticas agregadas, prontas para consumo (dashboard / DW),
           respondendo diretamente às perguntas de negócio.

Saída: arquivos .csv/.parquet em data/bronze, data/prata e data/ouro.
Esses arquivos são depois enviados ao Data Lake (MinIO/S3) por 03_upload_minio.py.

Disciplina: Sistemas de Apoio à Decisão — Trabalho AV2
"""

import glob
import os
import re
import unicodedata
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
UPLOADS = os.environ.get("UPLOADS_DIR", os.path.join(BASE, "raw"))

for camada in ("bronze", "prata", "ouro"):
    os.makedirs(os.path.join(DATA, camada), exist_ok=True)


# --------------------------------------------------------------------------- #
# BRONZE — ingestão crua
# --------------------------------------------------------------------------- #
def construir_bronze() -> pd.DataFrame:
    # Aceita tanto os nomes "06__Interna*.csv" quanto os do portal oficial
    # ("06. Internações Hospitalares (Referencia_ ...).csv"), removendo duplicatas.
    padroes = ["06*Interna*.csv", "06*interna*.csv"]
    arquivos = []
    for pad in padroes:
        arquivos.extend(glob.glob(os.path.join(UPLOADS, pad)))
    arquivos = sorted(set(arquivos))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV de internação encontrado em {UPLOADS}. "
            "Defina UPLOADS_DIR ou coloque os CSVs em ./raw/"
        )
    partes = []
    for f in arquivos:
        d = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)
        d["arquivo_origem"] = os.path.basename(f)
        partes.append(d)
    bronze = pd.concat(partes, ignore_index=True)
    bronze.columns = ["data_internacao", "especialidade", "municipio",
                      "idade", "sexo", "arquivo_origem"]
    bronze.to_csv(os.path.join(DATA, "bronze", "internacoes_bronze.csv"), index=False)
    print(f"[BRONZE] {len(arquivos)} arquivos -> {len(bronze):,} registros crus")
    return bronze


# --------------------------------------------------------------------------- #
# PRATA — limpeza e padronização
# --------------------------------------------------------------------------- #
def _titulo(texto: str) -> str:
    """Title-case respeitando preposições (São Gonçalo do Amarante)."""
    minus = {"de", "da", "do", "das", "dos", "e"}
    palavras = texto.lower().split()
    out = [p if p in minus else p.capitalize() for p in palavras]
    if out:
        out[0] = out[0].capitalize()
    return " ".join(out)


# Mapa de unificação de especialidades (agrupa variantes na mesma área-mãe)
def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper().strip())


REGRAS_ESPECIALIDADE = [
    ("ONCOLOGIA", "Oncologia"),
    ("TRANSPLANTE", "Transplante"),
    ("CARDIO", "Cardiologia"),
    ("NEURO", "Neurologia/Neurocirurgia"),
    ("UROLOGIA", "Urologia"),
    ("GASTRO", "Gastroenterologia"),
    ("PEDIATRIA", "Pediatria"),
    ("OTORRINO", "Otorrinolaringologia"),
    ("VASCULAR", "Cirurgia Vascular"),
    ("CABECA E PESCOCO", "Cirurgia de Cabeça e Pescoço"),
    ("APARELHO DIGESTIVO", "Cirurgia do Aparelho Digestivo"),
    ("BARIATRICA", "Cirurgia Bariátrica"),
    ("CIRURGIA GERAL", "Cirurgia Geral"),
    ("CLINICA GERAL", "Clínica Geral"),
    ("CLINICA MEDICA", "Clínica Médica"),
    ("ORTOPEDIA", "Ortopedia"),
    ("GINECOLOGIA", "Ginecologia/Obstetrícia"),
    ("OBSTETRICIA", "Ginecologia/Obstetrícia"),
    ("NEFROLOGIA", "Nefrologia"),
    ("PNEUMO", "Pneumologia"),
    ("ENDOCRINO", "Endocrinologia"),
    ("HEMATOLOGIA", "Hematologia"),
    ("PLASTICA", "Cirurgia Plástica"),
    ("TORACICA", "Cirurgia Torácica"),
    ("PROCTOLOGIA", "Proctologia"),
    ("MASTOLOGIA", "Mastologia"),
    ("OFTALMO", "Oftalmologia"),
    ("REUMATO", "Reumatologia"),
    ("INFECTO", "Infectologia"),
]


def unificar_especialidade(valor: str) -> str:
    if pd.isna(valor):
        return "Não informado"
    chave = _norm_key(valor)
    for padrao, nome in REGRAS_ESPECIALIDADE:
        if padrao in chave:
            return nome
    return _titulo(valor)


def faixa_etaria(idade) -> str:
    if pd.isna(idade):
        return "Não informado"
    i = int(idade)
    faixas = [(0, 12, "0-12 (Criança)"), (13, 17, "13-17 (Adolescente)"),
              (18, 29, "18-29 (Jovem adulto)"), (30, 44, "30-44 (Adulto)"),
              (45, 59, "45-59 (Meia-idade)"), (60, 74, "60-74 (Idoso)"),
              (75, 200, "75+ (Idoso longevo)")]
    for lo, hi, rot in faixas:
        if lo <= i <= hi:
            return rot
    return "Não informado"


def construir_prata(bronze: pd.DataFrame) -> pd.DataFrame:
    p = bronze.copy()

    # Datas: dd/mm/aaaa HH:MM -> datetime; descarta inválidas
    p["data_internacao"] = pd.to_datetime(
        p["data_internacao"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    invalidas = p["data_internacao"].isna().sum()
    p = p.dropna(subset=["data_internacao"])

    # Idade: numérica e plausível (0–120)
    p["idade"] = pd.to_numeric(p["idade"], errors="coerce")
    p = p[(p["idade"] >= 0) & (p["idade"] <= 120)]

    # Sexo: normaliza F/M/variações
    mapa_sexo = {"FEMININO": "Feminino", "F": "Feminino",
                 "MASCULINO": "Masculino", "M": "Masculino"}
    p["sexo"] = p["sexo"].astype(str).str.upper().str.strip().map(mapa_sexo)
    p = p.dropna(subset=["sexo"])

    # Especialidade unificada e município padronizado
    p["especialidade"] = p["especialidade"].apply(unificar_especialidade)
    p["municipio"] = p["municipio"].astype(str).apply(_titulo)

    # Derivações para análise
    p["faixa_etaria"] = p["idade"].apply(faixa_etaria)
    p["ano"] = p["data_internacao"].dt.year
    p["mes"] = p["data_internacao"].dt.month
    p["ano_mes"] = p["data_internacao"].dt.to_period("M").astype(str)
    p["e_natal"] = np.where(p["municipio"] == "Natal", "Natal", "Interior/Outros")

    p = p.drop(columns=["arquivo_origem"]).reset_index(drop=True)
    p.to_csv(os.path.join(DATA, "prata", "internacoes_prata.csv"), index=False)
    print(f"[PRATA]  {len(p):,} registros limpos "
          f"({invalidas} data(s) inválida(s) removida(s); "
          f"{p['especialidade'].nunique()} especialidades após unificação)")
    return p


# --------------------------------------------------------------------------- #
# OURO — agregações analíticas (respondem às perguntas de negócio)
# --------------------------------------------------------------------------- #
def construir_ouro(prata: pd.DataFrame) -> dict:
    saidas = {}

    # P1 — especialidades com mais internações
    saidas["ouro_por_especialidade"] = (
        prata.groupby("especialidade").size()
        .reset_index(name="internacoes").sort_values("internacoes", ascending=False)
    )
    # P2 — perfil etário e de gênero
    saidas["ouro_por_faixa_sexo"] = (
        prata.groupby(["faixa_etaria", "sexo"]).size()
        .reset_index(name="internacoes")
    )
    # P3 — padrões por município
    saidas["ouro_por_municipio"] = (
        prata.groupby("municipio").size()
        .reset_index(name="internacoes").sort_values("internacoes", ascending=False)
    )
    saidas["ouro_natal_vs_interior"] = (
        prata.groupby("e_natal").size().reset_index(name="internacoes")
    )
    # P4 — sazonalidade
    saidas["ouro_sazonalidade"] = (
        prata.groupby("ano_mes").size().reset_index(name="internacoes")
        .sort_values("ano_mes")
    )
    saidas["ouro_sazonalidade_mes"] = (
        prata.groupby(["mes", "ano"]).size().reset_index(name="internacoes")
    )

    for nome, df in saidas.items():
        df.to_csv(os.path.join(DATA, "ouro", f"{nome}.csv"), index=False)
    print(f"[OURO]   {len(saidas)} tabelas analíticas geradas")
    return saidas


if __name__ == "__main__":
    print("=" * 60)
    print("ETL DATA LAKE — INTERNAÇÕES HUOL/UFRN")
    print("=" * 60)
    bronze = construir_bronze()
    prata = construir_prata(bronze)
    ouro = construir_ouro(prata)
    print("\nResumo das perguntas de negócio:")
    print("  Top 3 especialidades:")
    print(ouro["ouro_por_especialidade"].head(3).to_string(index=False))
    print("\n  Natal vs Interior:")
    print(ouro["ouro_natal_vs_interior"].to_string(index=False))
    print("\nETL concluído com sucesso.")
