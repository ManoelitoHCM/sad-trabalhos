"""
run_all.py — Orquestra o pipeline completo do Trabalho AV2 (exceto envio ao MinIO).

Executa, em ordem: ETL das internações, geração e análise dos comentários,
construção do Data Warehouse e geração dos gráficos.

Uso:  python scripts/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ,
           UPLOADS_DIR=os.environ.get("UPLOADS_DIR", os.path.join(HERE, "..", "raw")))

ETAPAS = [
    "01_etl_internacoes.py",
    "02a_gerar_comentarios.py",
    "02b_analise_sentimento.py",
    "03_data_warehouse.py",
    "04_visualizacao.py",
]

for e in ETAPAS:
    print(f"\n{'=' * 60}\n>>> {e}\n{'=' * 60}")
    r = subprocess.run([sys.executable, os.path.join(HERE, e)], env=ENV)
    if r.returncode != 0:
        sys.exit(f"Falha em {e}")

print("\nPipeline completo.")
print("Para enviar ao Data Lake (MinIO): python scripts/05_upload_minio.py")
