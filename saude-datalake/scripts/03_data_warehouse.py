"""
03_data_warehouse.py
====================
Modelagem do Data Warehouse derivado da camada OURO, em esquema estrela.

Modelo dimensional:

  DIM_TEMPO        (sk_tempo, data, ano, mes, nome_mes, trimestre, ano_mes)
  DIM_ESPECIALIDADE(sk_especialidade, especialidade)
  DIM_MUNICIPIO    (sk_municipio, municipio, e_natal)
  DIM_PACIENTE     (sk_paciente, sexo, faixa_etaria)
  -----------------------------------------------------------------
  FATO_INTERNACAO  (sk_tempo, sk_especialidade, sk_municipio,
                    sk_paciente, qtd_internacoes)

Grão da fato: uma linha por combinação (tempo × especialidade × município ×
perfil de paciente), com a contagem de internações como medida aditiva.

Saída: data/dw/*.csv (uma tabela por arquivo).
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
DW = os.path.join(DATA, "dw")
os.makedirs(DW, exist_ok=True)

MESES = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio",
         6: "Junho", 7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro",
         11: "Novembro", 12: "Dezembro"}


def construir_dw():
    prata = pd.read_csv(os.path.join(DATA, "prata", "internacoes_prata.csv"))
    prata["data"] = pd.to_datetime(prata["data_internacao"]).dt.date

    # ---- Dimensão Tempo ----
    dim_tempo = (prata[["data", "ano", "mes", "ano_mes"]].drop_duplicates()
                 .sort_values("data").reset_index(drop=True))
    dim_tempo.insert(0, "sk_tempo", range(1, len(dim_tempo) + 1))
    dim_tempo["nome_mes"] = dim_tempo["mes"].map(MESES)
    dim_tempo["trimestre"] = ((dim_tempo["mes"] - 1) // 3 + 1).map(lambda q: f"T{q}")

    # ---- Dimensão Especialidade ----
    dim_esp = pd.DataFrame({"especialidade": sorted(prata["especialidade"].unique())})
    dim_esp.insert(0, "sk_especialidade", range(1, len(dim_esp) + 1))

    # ---- Dimensão Município ----
    dim_mun = (prata[["municipio", "e_natal"]].drop_duplicates()
               .sort_values("municipio").reset_index(drop=True))
    dim_mun.insert(0, "sk_municipio", range(1, len(dim_mun) + 1))

    # ---- Dimensão Paciente (perfil) ----
    dim_pac = (prata[["sexo", "faixa_etaria"]].drop_duplicates()
               .sort_values(["sexo", "faixa_etaria"]).reset_index(drop=True))
    dim_pac.insert(0, "sk_paciente", range(1, len(dim_pac) + 1))

    # ---- Fato ----
    fato = prata.merge(dim_tempo[["sk_tempo", "data"]], on="data") \
                .merge(dim_esp, on="especialidade") \
                .merge(dim_mun[["sk_municipio", "municipio"]], on="municipio") \
                .merge(dim_pac, on=["sexo", "faixa_etaria"])
    fato_agg = (fato.groupby(["sk_tempo", "sk_especialidade", "sk_municipio", "sk_paciente"])
                .size().reset_index(name="qtd_internacoes"))

    # Persistência
    dim_tempo.to_csv(os.path.join(DW, "dim_tempo.csv"), index=False)
    dim_esp.to_csv(os.path.join(DW, "dim_especialidade.csv"), index=False)
    dim_mun.to_csv(os.path.join(DW, "dim_municipio.csv"), index=False)
    dim_pac.to_csv(os.path.join(DW, "dim_paciente.csv"), index=False)
    fato_agg.to_csv(os.path.join(DW, "fato_internacao.csv"), index=False)

    print("Data Warehouse (esquema estrela) construído:")
    print(f"  dim_tempo        : {len(dim_tempo):>6,} linhas")
    print(f"  dim_especialidade: {len(dim_esp):>6,} linhas")
    print(f"  dim_municipio    : {len(dim_mun):>6,} linhas")
    print(f"  dim_paciente     : {len(dim_pac):>6,} linhas")
    print(f"  fato_internacao  : {len(fato_agg):>6,} linhas "
          f"(soma = {fato_agg['qtd_internacoes'].sum():,} internações)")


if __name__ == "__main__":
    construir_dw()
