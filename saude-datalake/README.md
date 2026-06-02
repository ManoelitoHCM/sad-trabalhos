# Data Lake & Data Warehouse — Internações HUOL/UFRN (AV2)

Trabalho AV2 da disciplina **Sistemas de Apoio à Decisão**.

Hospital privado de Natal (RN) que estuda o HUOL/UFRN — hospital público de referência —
para embasar sua decisão de **credenciamento ao SUS**. O projeto integra **dados
estruturados** (internações do SUS) e **não estruturados** (comentários do Instagram)
em um **Data Lake**, deriva um **Data Warehouse** dimensional e apresenta os insights.

## Arquitetura

```
CSVs SUS ─┐
          ├─► BRONZE ──► PRATA ──► OURO ──► Data Warehouse (estrela)
Instagram ┘   (cru)     (limpo)   (analítico)        │
                                       │             └─► gráficos (matplotlib)
                                       └─► MinIO (S3) via Docker
```

## Estrutura
| Caminho | Descrição |
|---|---|
| `scripts/01_etl_internacoes.py` | ETL bronze→prata→ouro das internações |
| `scripts/02a_gerar_comentarios.py` | Gera base sintética de comentários |
| `scripts/02b_analise_sentimento.py` | Limpeza, tokenização e sentimento |
| `scripts/02c_pipeline_graph_api.md` | Pipeline real da Instagram Graph API (documentado) |
| `scripts/03_data_warehouse.py` | Data Warehouse em esquema estrela |
| `scripts/04_visualizacao.py` | Gera os gráficos de insight |
| `scripts/05_upload_minio.py` | Envia as camadas ao Data Lake (MinIO/S3) |
| `scripts/run_all.py` | Roda todo o pipeline de uma vez |
| `docker-compose.yml` | Sobe o MinIO local |
| `Relatorio_AV2_DataLake_Saude.docx` | Texto detalhado do trabalho |

## Como rodar

```bash
pip install -r requirements.txt
# coloque os 8 CSVs de internação em ./raw/
python scripts/run_all.py            # ETL + sentimento + DW + gráficos
```

Para o Data Lake físico (MinIO):
```bash
docker compose up -d                 # sobe o MinIO (console: http://localhost:9001)
python scripts/05_upload_minio.py    # envia bronze/prata/ouro/dw ao bucket
```

## Perguntas de negócio (respostas)
1. **Especialidades com mais internações:** Cardiologia, Urologia, Gastroenterologia.
2. **Perfil:** ~equilíbrio entre sexos; 37% dos pacientes têm 60+ anos.
3. **Município:** ~35% Natal, ~65% interior/região metropolitana.
4. **Sazonalidade:** alta de +11,7% de 2024 para 2025.
5–6. **Sentimento (Instagram):** 38% positivo, 38% negativo (espera), 24% neutro.

## Dados
Internações: Portal de Dados Abertos — https://dados.gov.br/dados/conjuntos-dados/06-internacoes-hospitalares
Instagram: @huol_ufrn (comentários sintéticos; pipeline real documentado).
