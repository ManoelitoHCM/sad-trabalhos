# SAD ENCCEJA — Apoio à Decisão Educacional com K-NN

Trabalho AV1 da disciplina **Sistemas de Apoio à Tomada de Decisão** (Sistemas de Informação).

Sistema de Apoio à Decisão que usa o algoritmo **K-Nearest Neighbors (K-NN)** sobre os
microdados do **ENCCEJA 2024 (INEP)** para, a partir do **perfil socioeconômico** de um novo
candidato, estimar suas **notas esperadas**, o **risco de reprovação por área** e gerar
**recomendações pedagógicas** ao gestor de um cursinho preparatório.

## Como funciona
1. O gestor informa o perfil do candidato (certificação, faixa etária, sexo, UF, trabalho,
   renda familiar e escolaridade anterior).
2. O sistema encontra os `k` participantes históricos mais semelhantes (distância euclidiana
   sobre atributos normalizados, comparando apenas dentro do mesmo nível de certificação).
3. Calcula a nota esperada por área (média dos vizinhos) e o risco de reprovação
   (proporção de vizinhos aprovados), classificando o candidato em risco BAIXO/MÉDIO/ALTO.
4. Exibe a comparação com os vizinhos e recomendações automáticas.

## Estrutura
| Arquivo | Descrição |
|---|---|
| `knn_encceja.py` | Núcleo do K-NN (NumPy) + geração de recomendações |
| `app_gui.py` | Interface gráfica (Tkinter) |
| `preparar_dados.py` | Pipeline de higienização que gera a base a partir dos microdados |
| `encceja_2024_referencia.csv` | Base de referência (amostra estratificada de 45.000 participantes) |
| `Relatorio_ENCCEJA_KNN.docx` | Texto detalhando o processo de desenvolvimento |

## Execução
```bash
pip install -r requirements.txt
python app_gui.py
```
Para reconstruir a base a partir dos microdados oficiais:
```bash
python preparar_dados.py microdados_encceja_2024.zip
```

## Resultados (validação hold-out, Ensino Médio)
- Erro médio absoluto das notas previstas: **~12,5 pontos** (escala 0–180).
- Acurácia da previsão de aprovação: **~87%** com `k=15`, usando **apenas o perfil socioeconômico**.

## Dados
Microdados do ENCCEJA 2024 — INEP:
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja

