"""
app_gui.py
==========
Interface gráfica (Tkinter) do Sistema de Apoio à Decisão ENCCEJA / K-NN.

O gestor do cursinho preenche o PERFIL SOCIOECONÔMICO de um novo candidato e o
sistema retorna: notas esperadas por área, comparação com os vizinhos mais
próximos, risco de aprovação e recomendações pedagógicas.

Execução:  python app_gui.py
Requisitos: Python 3.10+, pandas, numpy, e o arquivo encceja_2024_referencia.csv
            na mesma pasta. (Tkinter já vem com o Python padrão.)
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

from knn_encceja import (
    ModeloKNN, gerar_recomendacoes, AREAS, AREAS_NOME,
    RENDA_LABELS, ESCOLARIDADE_LABELS, FAIXA_ETARIA_LABELS, UF_REGIAO,
)

UFS = sorted(UF_REGIAO.keys())
BASE = os.path.join(os.path.dirname(__file__), "encceja_2024_referencia.csv")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SAD ENCCEJA — Apoio à Decisão com K-NN")
        self.geometry("900x640")
        self.minsize(820, 600)

        try:
            self.modelo = ModeloKNN(BASE, k=15)
        except FileNotFoundError:
            messagebox.showerror("Erro", f"Base não encontrada:\n{BASE}")
            self.destroy()
            return

        self._montar_formulario()
        self._montar_saida()

    # ------------------------------------------------------------------ #
    def _montar_formulario(self):
        frm = ttk.LabelFrame(self, text="Perfil do candidato (dados da matrícula)")
        frm.pack(fill="x", padx=12, pady=10)

        self.var_cert = tk.StringVar(value="Ensino Médio")
        self.var_sexo = tk.StringVar(value="F")
        self.var_uf = tk.StringVar(value="CE")
        self.var_trab = tk.StringVar(value="Sim")
        self.var_faixa = tk.StringVar(value=FAIXA_ETARIA_LABELS[12])
        self.var_renda = tk.StringVar(value=RENDA_LABELS[1])
        self.var_esc = tk.StringVar(value=ESCOLARIDADE_LABELS[9])
        self.var_k = tk.IntVar(value=15)

        def combo(col, row, label, var, valores, width=24):
            ttk.Label(frm, text=label).grid(row=row, column=col*2, sticky="w",
                                            padx=6, pady=6)
            c = ttk.Combobox(frm, textvariable=var, values=valores,
                             state="readonly", width=width)
            c.grid(row=row, column=col*2 + 1, sticky="w", padx=6, pady=6)
            return c

        combo(0, 0, "Certificação:", self.var_cert,
              ["Ensino Fundamental", "Ensino Médio"])
        combo(1, 0, "Sexo:", self.var_sexo, ["F", "M"], width=8)
        combo(0, 1, "Faixa etária:", self.var_faixa,
              list(FAIXA_ETARIA_LABELS.values()))
        combo(1, 1, "UF:", self.var_uf, UFS, width=8)
        combo(0, 2, "Trabalha?", self.var_trab, ["Sim", "Não"], width=8)
        combo(1, 2, "Renda familiar:", self.var_renda,
              list(RENDA_LABELS.values()))
        combo(0, 3, "Escolaridade anterior:", self.var_esc,
              list(ESCOLARIDADE_LABELS.values()))
        ttk.Label(frm, text="k (vizinhos):").grid(row=1, column=6, sticky="w",
                                                  padx=6, pady=6)
        ttk.Spinbox(frm, from_=3, to=101, textvariable=self.var_k, width=6
                    ).grid(row=1, column=7, sticky="w", padx=6, pady=6)

        ttk.Button(frm, text="Analisar candidato", command=self.analisar
                   ).grid(row=2, column=0, columnspan=8, pady=10)

    def _montar_saida(self):
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        cols = ("area", "esperada", "faixa", "media", "aprov")
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings", height=6)
        for c, t, w in [("area", "Área", 200), ("esperada", "Nota esperada", 110),
                        ("faixa", "Faixa vizinhos", 130),
                        ("media", "vs média nível", 120),
                        ("aprov", "Aprov. vizinhos", 130)]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.column("area", anchor="w")
        self.tree.pack(fill="x")

        self.lbl_resumo = ttk.Label(wrap, text="", font=("", 11, "bold"))
        self.lbl_resumo.pack(anchor="w", pady=(10, 4))

        self.txt = tk.Text(wrap, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")

    # ------------------------------------------------------------------ #
    def _coletar_perfil(self) -> dict:
        inv = lambda d: {v: k for k, v in d.items()}
        return dict(
            certificacao=1 if self.var_cert.get() == "Ensino Fundamental" else 2,
            sexo=self.var_sexo.get(),
            uf=self.var_uf.get(),
            trabalha=self.var_trab.get(),
            faixa_etaria=inv(FAIXA_ETARIA_LABELS)[self.var_faixa.get()],
            renda_ord=inv(RENDA_LABELS)[self.var_renda.get()],
            escolaridade_ord=inv(ESCOLARIDADE_LABELS)[self.var_esc.get()],
        )

    def analisar(self):
        perfil = self._coletar_perfil()
        try:
            r = self.modelo.prever(perfil, k=self.var_k.get())
        except Exception as e:
            messagebox.showerror("Erro na análise", str(e))
            return

        # Tabela por área
        for i in self.tree.get_children():
            self.tree.delete(i)
        for a in AREAS:
            d = r["areas"][a]
            self.tree.insert("", "end", values=(
                AREAS_NOME[a],
                f"{d['nota_esperada']:.0f}",
                f"{d['nota_min_viz']:.0f}–{d['nota_max_viz']:.0f}",
                f"{d['vs_media']:+.0f}",
                f"{d['taxa_aprovacao_viz']*100:.0f}%",
            ))

        cor = {"BAIXO": "#1a7f37", "MÉDIO": "#b58900", "ALTO": "#c0392b"}[r["nivel_risco"]]
        self.lbl_resumo.configure(
            text=(f"Risco de reprovação: {r['nivel_risco']}   |   "
                  f"Aprovação esperada: ~{r['n_areas_esperadas']:.1f}/4 áreas   |   "
                  f"Redação esperada: {r['nota_redacao_esperada']:.1f}/10   "
                  f"(k={r['k']} vizinhos)"),
            foreground=cor,
        )

        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "RECOMENDAÇÕES AO GESTOR\n", )
        for rec in gerar_recomendacoes(r):
            self.txt.insert("end", f"  • {rec}\n")
        self.txt.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
