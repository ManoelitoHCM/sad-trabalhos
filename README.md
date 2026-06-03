# SAD — Trabalhos de Sistemas de Apoio à Decisão

Monorepo com os trabalhos da disciplina **Sistemas de Apoio à Tomada de Decisão** (Sistemas de Informação). Cada projeto aplica uma técnica diferente de apoio à decisão a uma base de dados pública real, com código, relatório slides.

## Trabalhos

| # | Projeto | Técnica | Base de dados | Objetivo |
|---|---|---|---|---|
| 1 | **ENCCEJA** | K-NN (do zero, NumPy) | Microdados ENCCEJA 2024 (INEP) | Estimar notas e risco de reprovação de candidatos a partir do perfil socioeconômico |
| 2 | **Trânsito** | Árvore de Decisão (entropia) | Acidentes da PRF 2025 | Identificar fatores ligados à ocorrência de vítimas e priorizar campanhas |
| 3 | **Saúde** | Data Lake + Data Warehouse | Internações do SUS + comentários (sentimento) | Consolidar dados estruturados e não estruturados para análise gerencial |

## Estrutura do repositório

```
sad-trabalhos/
├── encceja-knn/                ← Trabalho 1
├── transito-arvore-decisao/    ← Trabalho 2
├── saude-datalake/             ← Trabalho 3
```
## Requisitos gerais

- **Python 3.10+**. Cada projeto tem seu próprio `requirements.txt` (`pip install -r requirements.txt`).
- O Trabalho 3 também exige **Docker Desktop** (para o MinIO).
- As **bases de dados** (zips e CSVs grandes) **não estão versionadas** no Git (ver `.gitignore`); baixe-as nos portais oficiais indicados em cada seção.

---

## 1. ENCCEJA — Sistema de Apoio à Decisão com K-NN

Sistema que usa **K-Nearest Neighbors** sobre os microdados do **ENCCEJA 2024** para, a partir do **perfil socioeconômico** de um novo candidato de um cursinho preparatório, estimar suas **notas esperadas por área**, o **risco de reprovação** e gerar **recomendações pedagógicas** ao gestor.

**Como funciona**
1. O gestor informa o perfil do candidato (certificação, faixa etária, sexo, UF, trabalho, renda familiar e escolaridade anterior).
2. O sistema encontra os `k` participantes históricos mais semelhantes (distância euclidiana sobre atributos normalizados, comparando **apenas dentro do mesmo nível de certificação**).
3. Calcula a nota esperada por área (média dos vizinhos) e o risco de reprovação (proporção de vizinhos aprovados), classificando em **BAIXO / MÉDIO / ALTO**.
4. Exibe a comparação com os vizinhos e recomendações automáticas.

O K-NN é implementado **do zero em NumPy** (sem caixa-preta), com **normalização min-max**, ponderação das regiões (1/√2) e filtragem por nível de certificação.

**Arquivos**

| Arquivo | Descrição |
|---|---|
| `knn_encceja.py` | Núcleo do K-NN (NumPy) + geração de recomendações |
| `app_gui.py` | Interface gráfica (Tkinter) |
| `preparar_dados.py` | Pipeline de higienização que gera a base a partir dos microdados |
| `encceja_2024_referencia.csv` | Base de referência (amostra estratificada de ~45.000 participantes) |
| `Relatorio_ENCCEJA_KNN.docx` | Relatório do processo de desenvolvimento |
| `Apresentacao_ENCCEJA_KNN.pptx` | Slides do vídeo |

**Execução**
```bash
pip install -r requirements.txt
python app_gui.py
```
Para reconstruir a base a partir dos microdados oficiais:
```bash
python preparar_dados.py microdados_encceja_2024.zip
```

**Resultados (validação hold-out, Ensino Médio)**
- Erro médio absoluto das notas previstas: **~12,5 pontos** (escala 0–180).
- Acurácia da previsão de aprovação: **~87%** com `k=15`, usando **apenas o perfil socioeconômico**.

**Dados:** Microdados do ENCCEJA 2024 — INEP — <https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja>

---

## 2. Trânsito — Árvore de Decisão (acidentes da PRF)

Aplica uma **Árvore de Decisão** (critério de **entropia**, `max_depth=6`) aos dados de **acidentes da Polícia Rodoviária Federal (2025)** para identificar os atributos mais associados à **ocorrência de vítimas** e apoiar a priorização de campanhas educativas de trânsito.

**Como funciona**
1. Pré-processamento dos registros de acidentes (seleção e codificação dos atributos relevantes: causa, tipo de acidente, fase do dia, condição da via, etc.).
2. Treino da árvore com critério de entropia e profundidade controlada (`max_depth=6`) para manter o modelo interpretável.
3. Análise da **importância dos atributos** e comparação com uma **linha de base** (sempre prever a classe majoritária).
4. Versão com `class_weight='balanced'` para melhorar a detecção da classe minoritária ("Sem Vítimas").

**Arquivos**

| Arquivo | Descrição |
|---|---|
| `arvore_decisao_transito.py` | Treino da árvore, avaliação e geração dos gráficos |
| `texto_analise.md` | Texto com a análise do processo e dos resultados |
| `feature_importance.png` | Gráfico da importância dos atributos |
| `Apresentacao_Transito_ArvoreDecisao.pptx` | Slides do vídeo |

**Execução**
```bash
pip install -r requirements.txt
python arvore_decisao_transito.py
```

**Resultados (≈72.529 registros)**
- Ganho de **+2,1 p.p.** de acurácia sobre a linha de base (classe majoritária).
- O modelo balanceado eleva o *recall* de "Sem Vítimas" de **0,14 → 0,50**, ao custo de um pouco de acurácia global — um trade-off discutido no relatório.

**Dados:** Dados Abertos da PRF — Acidentes 2025 — <https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf>

---

## 3. Saúde — Data Lake + Data Warehouse (SUS + sentimento)

Pipeline de **Data Lake** que consolida dados **estruturados** (internações hospitalares do **SUS**) e **não estruturados** (comentários de redes sociais, com **análise de sentimento**), modelando um **Data Warehouse** em esquema estrela e usando **MinIO/Docker** como solução gratuita de Data Lake.

**Arquitetura**
- **Camadas (medalhão):** `bronze` (dado bruto) → `prata` (limpo e padronizado) → `ouro` (agregado para análise).
- **Data Warehouse (esquema estrela):** tabela **fato** de internações + 4 **dimensões** — tempo (quando), especialidade (o quê), município (onde) e paciente (sexo/faixa etária). A soma da fato bate com a camada prata (integridade verificada).
- **Sentimento (por léxico):** classificação positivo/negativo/neutro por dicionário de palavras — leve e transparente. Os comentários são **sintéticos** (a Graph API do Instagram só libera comentários para administradores da página); o pipeline real de coleta está documentado em `02c_pipeline_graph_api.md`.
- **Infraestrutura:** **MinIO** (armazenamento de objetos compatível com S3) subido via **Docker**.

**Estrutura**
```
saude-datalake/
├── scripts/          ← pipeline (.py) + 02c_pipeline_graph_api.md
├── raw/              ← os 8 CSVs trimestrais de internação
├── docker-compose.yml
├── requirements.txt
├── Relatorio_AV2_DataLake_Saude.docx
└── Apresentacao_AV2_DataLake_Saude.pptx
```
As pastas `data/` (bronze/prata/ouro/dw) e `figuras/` são **geradas pelos scripts**.

O `scripts/run_all.py` orquestra o pipeline em sequência: ETL das internações (`01_etl_internacoes.py`), geração dos comentários, análise de sentimento, construção do Data Warehouse e geração das **6 visualizações**. O envio das camadas para o MinIO é feito por `05_upload_minio.py`.

**Execução**
```bash
pip install -r requirements.txt          # pandas, numpy, matplotlib, boto3
python scripts/run_all.py                 # gera data/ e figuras/

# Sobe o Data Lake (com o Docker Desktop aberto):
docker compose up -d
python scripts/05_upload_minio.py         # envia bronze/prata/ouro ao MinIO
```
Console do MinIO: <http://localhost:9001> (usuário `minioadmin` / senha `minioadmin`).

**Resultados (16.044 internações)**
- 37 especialidades; 51%/49% por sexo; ~37% idosos; 35%/65% capital (Natal) vs. interior; **+11,7%** de 2024 → 2025.
- Sentimento dos comentários: **38% positivo / 38% negativo / 24% neutro**, com 97,0% de concordância do léxico.

**Dados:** Internações hospitalares do SUS (DATASUS / Ministério da Saúde) + base sintética de comentários.

---

## Autoria

**Manoelito Holanda** — Sistemas de Informação · Disciplina de Sistemas de Apoio à Tomada de Decisão.