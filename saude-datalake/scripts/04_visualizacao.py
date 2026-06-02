"""
04_visualizacao.py
==================
Gera os gráficos de insight a partir da camada OURO do Data Lake, respondendo
visualmente às perguntas de negócio do trabalho (estruturadas + sentimento).

Lê de   : data/ouro/*.csv
Salva em: figuras/*.png

Gráficos gerados:
  fig1_especialidades.png      P1 — especialidades com mais internações (top 12)
  fig2_perfil_faixa_sexo.png   P2 — perfil etário e de gênero
  fig3_municipios.png          P3 — internações por município + Natal vs Interior
  fig4_sazonalidade.png        P4 — evolução mensal 2024–2025
  fig5_sentimento.png          P6 — distribuição de sentimento dos comentários
  fig6_palavras.png            P5 — palavras mais recorrentes por sentimento

Disciplina: Sistemas de Apoio à Decisão — Trabalho AV2
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURO = os.path.join(BASE, "data", "ouro")
FIG = os.path.join(BASE, "figuras")
os.makedirs(FIG, exist_ok=True)

# Paleta consistente (tema saúde: azul-petróleo + acentos)
AZUL = "#1F6F8B"
AZUL_CLARO = "#5BA4B8"
VERDE = "#2E8B57"
AMBAR = "#E8A100"
VERMELHO = "#C0392B"
CINZA = "#94A3B8"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold"})


def _ler(nome):
    return pd.read_csv(os.path.join(OURO, nome))


def _rotular_barh(ax, valores, fmt="{:,.0f}"):
    for i, v in enumerate(valores):
        ax.text(v, i, "  " + fmt.format(v).replace(",", "."), va="center", fontsize=10)


# --------------------------------------------------------------------------- #
def fig_especialidades():
    df = _ler("ouro_por_especialidade.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = [VERMELHO if i >= 10 else AZUL for i in range(len(df))]
    ax.barh(df["especialidade"], df["internacoes"], color=cores)
    ax.set_title("P1 — Especialidades com mais internações (Top 12)")
    ax.set_xlabel("Nº de internações")
    ax.set_xlim(0, df["internacoes"].max() * 1.12)
    _rotular_barh(ax, df["internacoes"].values)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_especialidades.png"), dpi=150)
    plt.close()
    print("  fig1_especialidades.png")


def fig_perfil():
    df = _ler("ouro_por_faixa_sexo.csv")
    ordem = ["0-12 (Criança)", "13-17 (Adolescente)", "18-29 (Jovem adulto)",
             "30-44 (Adulto)", "45-59 (Meia-idade)", "60-74 (Idoso)",
             "75+ (Idoso longevo)"]
    piv = df.pivot_table(index="faixa_etaria", columns="sexo",
                         values="internacoes", aggfunc="sum").reindex(ordem).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(piv))
    fem = piv.get("Feminino", pd.Series([0] * len(piv), index=piv.index))
    mas = piv.get("Masculino", pd.Series([0] * len(piv), index=piv.index))
    ax.barh([i + 0.2 for i in y], fem, height=0.4, label="Feminino", color=VERMELHO)
    ax.barh([i - 0.2 for i in y], mas, height=0.4, label="Masculino", color=AZUL)
    ax.set_yticks(list(y))
    ax.set_yticklabels(piv.index)
    ax.set_title("P2 — Perfil etário e de gênero dos pacientes")
    ax.set_xlabel("Nº de internações")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_perfil_faixa_sexo.png"), dpi=150)
    plt.close()
    print("  fig2_perfil_faixa_sexo.png")


def fig_municipios():
    df = _ler("ouro_por_municipio.csv").head(10).iloc[::-1]
    nat = _ler("ouro_natal_vs_interior.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    ax1.barh(df["municipio"], df["internacoes"], color=AZUL)
    ax1.set_title("P3 — Internações por município (Top 10)")
    ax1.set_xlabel("Nº de internações")
    ax1.set_xlim(0, df["internacoes"].max() * 1.15)
    _rotular_barh(ax1, df["internacoes"].values)

    cores = [AZUL if e == "Natal" else AMBAR for e in nat["e_natal"]]
    ax2.pie(nat["internacoes"], labels=nat["e_natal"], autopct="%1.1f%%",
            colors=cores, startangle=90, textprops={"fontsize": 11})
    ax2.set_title("Natal vs Interior/Outros")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_municipios.png"), dpi=150)
    plt.close()
    print("  fig3_municipios.png")


def fig_sazonalidade():
    df = _ler("ouro_sazonalidade.csv").sort_values("ano_mes")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df["ano_mes"], df["internacoes"], marker="o", color=AZUL, linewidth=2.2)
    # marca a virada de ano
    if "2025-01" in set(df["ano_mes"]):
        idx = list(df["ano_mes"]).index("2025-01")
        ax.axvline(idx - 0.5, color=CINZA, linestyle=":", linewidth=1.2)
        ax.text(idx - 0.4, ax.get_ylim()[1] * 0.96, "2025", color=CINZA, fontsize=10)
    ax.set_title("P4 — Sazonalidade das internações (2024–2025)")
    ax.set_ylabel("Nº de internações")
    ax.set_xticks(range(0, len(df), 2))
    ax.set_xticklabels(df["ano_mes"].iloc[::2], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_sazonalidade.png"), dpi=150)
    plt.close()
    print("  fig4_sazonalidade.png")


def fig_sentimento():
    df = _ler("ouro_sentimento.csv")
    ordem = {"positivo": (VERDE, "Positivo"), "negativo": (VERMELHO, "Negativo"),
             "neutro": (CINZA, "Neutro")}
    df = df.set_index("sentimento").reindex(list(ordem)).reset_index()
    cores = [ordem[s][0] for s in df["sentimento"]]
    rot = [ordem[s][1] for s in df["sentimento"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    total = df["quantidade"].sum()
    wedges, _, _ = ax.pie(df["quantidade"], labels=rot, colors=cores,
                          autopct=lambda p: f"{p:.0f}%\n({int(round(p*total/100))})",
                          startangle=90, textprops={"fontsize": 12})
    ax.set_title("P6 — Sentimento dos comentários do Instagram")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_sentimento.png"), dpi=150)
    plt.close()
    print("  fig5_sentimento.png")


def fig_palavras():
    df = _ler("ouro_palavras_frequentes.csv")
    grupos = [("positivo", VERDE), ("negativo", VERMELHO), ("neutro", CINZA)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (sent, cor) in zip(axes, grupos):
        sub = df[df["sentimento"] == sent].head(8).iloc[::-1]
        ax.barh(sub["palavra"], sub["frequencia"], color=cor)
        ax.set_title(f"P5 — Termos: {sent}")
        ax.set_xlabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_palavras.png"), dpi=150)
    plt.close()
    print("  fig6_palavras.png")


if __name__ == "__main__":
    print("Gerando gráficos de insight na pasta figuras/:")
    fig_especialidades()
    fig_perfil()
    fig_municipios()
    fig_sazonalidade()
    fig_sentimento()
    fig_palavras()
    print("Visualização concluída.")
