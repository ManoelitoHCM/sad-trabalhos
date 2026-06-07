# Seleção de Bolsistas com Algoritmo Genético — Microdados do ENEM

Trabalho da disciplina **Sistemas de Apoio à Tomada de Decisão** (Sistemas de Informação).

Uma fundação educacional privada quer conceder bolsas a estudantes de alto potencial do ENEM. O sistema seleciona automaticamente o **melhor grupo de 100 bolsistas** usando um **algoritmo genético** que otimiza, ao mesmo tempo, três objetivos:

1. **Performance acadêmica** — média das 5 notas (peso **0.50**)
2. **Diversidade** — índice de diversidade (peso **0.30**)
3. **Cobertura geográfica nacional** — UFs representadas (peso **0.20**)

---

## Estrutura do projeto

| Arquivo | Descrição |
|---|---|
| `algoritmo_genetico_bolsas.py` | Algoritmo genético completo: dados, aptidão, operadores, gráficos, relatório. |
| `interface_bolsas.py` | **Bônus** — interface gráfica (Tkinter) onde a fundação define os critérios. |
| `gerar_amostra_sintetica.py` | Gera amostra de teste no formato do ENEM. |
| `requirements.txt` | Dependências. |
| `roteiro_video.md` | Roteiro de narração para o vídeo de demonstração. |

---

## Instalação

```bash
pip install -r requirements.txt
```

Instala `pandas`, `numpy`, `matplotlib` e `python-docx`. Requer Python 3.10+.

> **Atenção:** a biblioteca do relatório é o **`python-docx`** (com hífen) — **não** `docx`. Se faltar, instale com `pip install python-docx`. O `tkinter` (da interface) já acompanha o Python na maioria das instalações (no Linux: `sudo apt install python3-tk`).

---

## ⚠️ Importante: microdados do ENEM 2024 e a diversidade por proxies

A partir de **2024**, o INEP passou a divulgar os microdados em **arquivos separados** (por anonimização / LGPD), dentro da pasta `DADOS/`:

- `RESULTADOS_2024.csv` — **notas** (`NU_NOTA_*`, incl. `NU_NOTA_REDACAO`), presença e dados da **escola** (dependência, localização) e da prova (`SG_UF_PROVA`). Chave: `NU_SEQUENCIAL`.
- `PARTICIPANTES_2024.csv` — **questionário socioeconômico** (`Q006` renda, `Q002` escolaridade da mãe) e **cor/raça**. Chave: `NU_INSCRICAO`.

As duas tabelas **não compartilham chave de ligação** — é impossível unir nota e perfil socioeconômico por participante (essa é justamente a proteção de privacidade). Por isso, ao usar os dados de 2024, este projeto trabalha **somente com o `RESULTADOS_2024.csv`** e mede a diversidade pelos atributos disponíveis nele:

- **Dependência administrativa da escola** (federal / estadual / municipal / privada) — proxy socioeconômico equivalente ao eixo "pública × privada".
- **Localização da escola** (urbana / rural).

A análise é restrita aos participantes que **declararam escola** (concluintes), para os quais esses atributos existem. O script entra nesse **"Modo Proxy 2024" automaticamente** ao detectar o `RESULTADOS_2024.csv`, e o relatório gerado já inclui uma observação metodológica explicando a limitação. Veja a seção *Limitação metodológica* abaixo.

---

## Como executar

**1) Com os microdados oficiais do ENEM 2024** (modo principal — só o `RESULTADOS`):

```bash
python algoritmo_genetico_bolsas.py --dados "microdados_enem_2024/DADOS/RESULTADOS_2024.csv" --amostra 100000
```

**2) Teste rápido com amostra sintética** (não precisa baixar a base):

```bash
python gerar_amostra_sintetica.py --n 5000 --saida amostra_enem.csv
python algoritmo_genetico_bolsas.py --dados amostra_enem.csv
```

**3) Alternativa — ENEM 2023** (arquivo único e vinculado, com o questionário **completo**: renda, escolaridade da mãe, tipo de escola e cor/raça):

```bash
python algoritmo_genetico_bolsas.py --dados MICRODADOS_ENEM_2023.csv --amostra 100000
```

**4) Interface gráfica (bônus):**

```bash
python interface_bolsas.py
```

### Argumentos
| Argumento | Descrição |
|---|---|
| `--dados` | Caminho do CSV (o `RESULTADOS_2024.csv`, a amostra sintética ou um arquivo único como o de 2023). |
| `--amostra` | Subamostra o pool para N candidatos (recomendado para a base real). |
| `--saida` | Pasta de saída (padrão: `saida`). |
| `--seed` | Semente aleatória (padrão: `42`, garante reprodutibilidade). |

> **Detecção automática:** o script identifica sozinho os nomes reais das colunas (a redação `NU_NOTA_REDACAO`, a UF `SG_UF_PROVA`), a codificação (UTF-8/latin-1) e o conjunto de atributos de diversidade disponível. Não é preciso configurar nada.

> **Memória/desempenho:** o `RESULTADOS_2024.csv` tem ~1,7 GB. O script lê apenas as colunas necessárias e filtra antes de processar, rodando bem em máquinas com 8 GB+ de RAM. A leitura inicial leva 1–2 minutos; o algoritmo em si, poucos segundos. Use `--amostra 100000` para um bom equilíbrio (100 mil candidatos já garantem diversidade e todas as 27 UFs).

---

## Saídas (pasta `saida/`)

- `grupo_ideal.csv` — os 100 bolsistas selecionados.
- `convergencia.png` — evolução da aptidão ao longo das gerações.
- `notas.png` — distribuição das notas do grupo.
- `diversidade.png` — composição do grupo nos atributos de diversidade.
- `distribuicao_geografica.png` — bolsistas por UF.
- `Relatorio_Bolsas_AG.docx` — relatório completo, gerado automaticamente com os resultados reais.

---

## Como funciona o algoritmo

- **Cromossomo:** um grupo de 100 candidatos (vetor de 100 índices distintos da base).
- **Aptidão:** `f = 0.50·notas + 0.30·diversidade + 0.20·cobertura`, com todas as componentes normalizadas em `[0, 1]`.
  - *Notas:* média das 5 notas ÷ 1000.
  - *Diversidade:* média da **entropia de Shannon normalizada** dos atributos disponíveis. Nos dados de **2024** (RESULTADOS): **dependência** e **localização** da escola. Em arquivo único com questionário (**ENEM ≤ 2023** ou amostra sintética): renda (Q006), escolaridade da mãe (Q002), tipo de escola e cor/raça.
  - *Cobertura:* fração das 27 UFs (26 estados + DF) presentes no grupo.
- **Operadores:** seleção por **torneio**, **crossover** por união dos pais + amostragem de 100 distintos, **mutação** trocando 5% dos genes, e **elitismo** (2 melhores preservados).
- **Parâmetros** (conforme enunciado): população **20**, gerações **100**, grupo **100**.

---

## Limitação metodológica (LGPD)

O enunciado cita renda, escolaridade da mãe, cor/raça e estado de residência. Com os **microdados de 2024**, parte desses atributos não pôde ser usada:

- **Renda, cor/raça e escolaridade da mãe** estão no `PARTICIPANTES`, **desvinculado** das notas — não há como associá-los a cada candidato. Foram substituídos pelos **proxies de escola** (dependência e localização).
- **Estado de residência** (`SG_UF_RESIDENCIA`) foi removido dos microdados desde 2020; usa-se a **UF da prova** (`SG_UF_PROVA`) como indicador geográfico — válido para o objetivo de cobertura nacional.

Essa limitação é inerente à anonimização dos dados públicos, não ao método. Para usar os atributos socioeconômicos completos, basta rodar o projeto sobre o **ENEM 2023** (arquivo único e vinculado).

---

## Resultados obtidos (ENEM 2024, amostra de 100.000 candidatos)

A partir de **4.332.944** registros, o pipeline manteve **1.565.797** participantes com escola declarada e **1.193.432** com todas as notas válidas, dos quais foram amostrados 100.000 para a otimização. Em 100 gerações:

| Métrica | Valor |
|---|---|
| Aptidão final | **0,7196** |
| Média geral das notas do grupo | **550,2** |
| Componente de diversidade (norm.) | **0,8151** |
| Cobertura geográfica | **27 / 27 UFs** |

A aptidão evoluiu de **0,588 → 0,720**. A cobertura nacional saturou (27/27) logo nas primeiras gerações e a diversidade subiu de **0,48 → 0,82**, enquanto a média das notas se manteve estável (~550). Isso evidencia o **trade-off multiobjetivo**: o algoritmo ganha diversidade e abrangência sem sacrificar o desempenho acadêmico — selecionar apenas as 100 maiores notas violaria os outros dois objetivos.

> Os números acima são reprodutíveis com `--seed 42`. Resultados podem variar com outra semente ou tamanho de amostra.
