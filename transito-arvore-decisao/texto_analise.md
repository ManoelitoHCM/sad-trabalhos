# Apoio à Decisão para Campanhas Educativas no Trânsito
## Análise por Árvore de Decisão — Base PRF 2025

**Disciplina:** Sistemas de Apoio à Tomada de Decisão  
**Curso:** Sistemas de Informação  

---

## 1. Introdução

O planejamento de campanhas educativas de trânsito historicamente se baseia em estatísticas gerais e experiência prática dos gestores. Este trabalho propõe uma abordagem orientada por dados, utilizando técnicas de aprendizado de máquina — especificamente o algoritmo de Árvore de Decisão baseado em entropia de Shannon e ganho de informação (ID3/C4.5) — para identificar os atributos mais associados à ocorrência de vítimas em sinistros registrados nas rodovias federais brasileiras.

A questão gerencial norteadora é: **quais características dos sinistros estão mais associadas à ocorrência de vítimas e devem ser priorizadas como foco de campanhas educativas?**

---

## 2. Base de Dados

A base utilizada é o arquivo *Sinistros de Trânsito Agrupados por Ocorrência — 2025*, disponibilizado pela Polícia Rodoviária Federal (PRF) no Portal de Dados Abertos do Governo Federal (dados.gov.br). O arquivo está no formato CSV com separador ponto-e-vírgula e codificação latin-1.

**Dimensões:** 72.529 registros × 30 atributos.

**Distribuição da variável `classificacao_acidente`:**

| Classificação          | Registros | Percentual |
|------------------------|-----------|------------|
| Com Vítimas Feridas    | 56.181    | 77,5%      |
| Sem Vítimas            | 11.138    | 15,4%      |
| Com Vítimas Fatais     |  5.209    |  7,2%      |
| Nulo (removido)        |      1    |    —       |

---

## 3. Preparação e Tratamento dos Dados

### 3.1 Criação da variável alvo

A variável alvo foi construída como **binária**, agrupando as categorias de classificação:

- **Classe 0 — Sem Vítimas:** sinistros sem feridos ou mortos.
- **Classe 1 — Com Vítimas:** sinistros com vítimas feridas ou fatais.

Essa binarização é adequada ao objetivo gerencial: identificar o que diferencia acidentes com consequências humanas dos que não têm.

### 3.2 Seleção de features

Foram selecionados **10 atributos contextuais** que descrevem as circunstâncias do sinistro — não seus resultados. Colunas como `mortos`, `feridos`, `ilesos` e `ignorados` foram **deliberadamente excluídas** para evitar vazamento de informação (*data leakage*): elas são consequências do sinistro, não causas, e incluí-las tornaria o modelo trivialmente preciso mas sem valor preditivo real.

| Feature                  | Descrição                                      |
|--------------------------|------------------------------------------------|
| `causa_acidente`         | Causa principal registrada pelo agente de PRF  |
| `tipo_acidente`          | Tipo de colisão ou evento (tombamento, capotamento etc.) |
| `sentido_via`            | Crescente ou decrescente                       |
| `fase_dia`               | Período do dia (pleno dia, noite, amanhecer etc.) |
| `uso_solo`               | Área urbana ou rural                           |
| `uf`                     | Estado onde o sinistro ocorreu                 |
| `tracado_via`            | Reta, curva, interseção, rotatória etc.        |
| `condicao_metereologica` | Chuva, céu claro, neblina etc.                 |
| `dia_semana`             | Dia da semana                                  |
| `tipo_pista`             | Simples, dupla ou múltipla                     |

### 3.3 Codificação

Como o scikit-learn exige entrada numérica, todas as variáveis categóricas foram convertidas com `LabelEncoder`. Valores nulos foram substituídos pela categoria `'Ignorado'` antes da codificação.

---

## 4. Implementação do Algoritmo

### 4.1 Fundamento teórico

O algoritmo utilizado é a **Árvore de Decisão com critério de entropia**, que implementa o conceito de **Ganho de Informação** do algoritmo ID3 (Iterative Dichotomiser 3), base do C4.5.

A **entropia de Shannon** mede a impureza de um conjunto *S*:

```
H(S) = - Σ p(c) · log₂ p(c)
```

onde `p(c)` é a proporção de exemplos da classe `c`. Quando todos os exemplos pertencem à mesma classe, `H = 0` (conjunto puro). Quando há distribuição igual entre classes, `H = 1`.

O **Ganho de Informação** de um atributo *A* sobre o conjunto *S* é:

```
IG(S, A) = H(S) - Σ (|Sv| / |S|) · H(Sv)
```

A cada nó, o algoritmo escolhe o atributo com maior ganho de informação — ou seja, aquele que mais reduz a incerteza sobre a classe alvo.

### 4.2 Hiperparâmetros

```python
DecisionTreeClassifier(
    criterion='entropy',  # Ganho de informação (não Gini)
    max_depth=6,          # Limita profundidade para evitar overfitting
    random_state=42       # Reprodutibilidade
)
```

A profundidade máxima de 6 foi escolhida para equilibrar capacidade preditiva e interpretabilidade, evitando que a árvore memorize ruído dos dados de treino.

---

## 5. Treinamento e Teste do Modelo

A base foi dividida em **70% treino** (50.770 registros) e **30% teste** (21.759 registros), com estratificação pela variável alvo para garantir proporções equivalentes em ambos os conjuntos.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

O modelo foi treinado exclusivamente nos dados de treino e avaliado nos dados de teste (nunca vistos durante o treinamento).

---

## 6. Avaliação do Modelo

### 6.1 Acurácia

O modelo alcançou **acurácia de 86,75%** no conjunto de teste. É importante, porém, comparar esse número com o baseline trivial como 84,6% dos sinistros têm vítimas, um classificador que sempre responde "Com Vítimas" já acertaria **84,6%**. O ganho do modelo sobre esse palpite é de apenas **+2,1 pontos percentuais**, e sua acurácia balanceada (média do recall das duas classes) é de apenas **0,57** — pouco acima de 0,50. Ou seja, a acurácia global é alta sobretudo porque a classe majoritária é fácil; o modelo ainda tem dificuldade em distinguir os sinistros sem vítimas. Isso não invalida a análise (o objetivo é identificar atributos relevantes, e estes são robustos), mas é uma leitura honesta da métrica.

### 6.2 Matriz de Confusão

|                        | Previsto: Sem Vítimas | Previsto: Com Vítimas |
|------------------------|----------------------:|----------------------:|
| **Real: Sem Vítimas**  |                   481 |                 2.860 |
| **Real: Com Vítimas**  |                    22 |                18.396 |

**Interpretação:**
- O modelo identifica corretamente **99,9%** dos sinistros com vítimas (recall = 1,00), o que é fundamental para não deixar casos graves sem atenção.
- A principal limitação é no recall da classe "Sem Vítimas" (14%): o modelo tende a classificar acidentes como "com vítimas" quando na verdade não há — um erro conservador e preferível ao inverso no contexto de saúde pública.
- O desequilíbrio de classes (84,6% com vítimas vs. 15,4% sem) influencia esse comportamento.

### 6.3 Relatório de Classificação

| Classe         | Precision | Recall | F1-Score | Support |
|----------------|----------:|-------:|---------:|--------:|
| Sem Vítimas    |      0,96 |   0,14 |     0,25 |   3.341 |
| Com Vítimas    |      0,87 |   1,00 |     0,93 |  18.418 |
| **Acurácia**   |           |        | **0,87** |  21.759 |
| Macro avg      |      0,91 |   0,57 |     0,59 |  21.759 |
| Weighted avg   |      0,88 |   0,87 |     0,82 |  21.759 |

### 6.4 Modelo alternativo com classes balanceadas

Para tratar o desbalanceamento (84,6% vs. 15,4%), treinou-se uma segunda árvore com `class_weight='balanced'`, que penaliza mais os erros na classe minoritária. O resultado evidencia o *trade-off* clássico:

| Métrica                       | Modelo padrão | Modelo balanceado |
|-------------------------------|--------------:|------------------:|
| Acurácia geral                |        86,75% |            74,40% |
| Acurácia balanceada           |          0,57 |              0,64 |
| Recall "Sem Vítimas"          |          0,14 |              0,50 |
| F1 macro                      |          0,59 |              0,61 |

O modelo balanceado sacrifica acurácia global, mas passa a detectar metade dos sinistros sem vítimas (recall 0,14 → 0,50) e melhora a acurácia balanceada. A escolha entre os dois depende do objetivo: para priorizar a detecção de casos graves o modelo padrão (recall ~1,0 em "Com Vítimas") é preferível; para uma análise equilibrada entre as classes, o balanceado é mais informativo. Em ambos, as features mais importantes permanecem `causa_acidente` e `tipo_acidente`, de modo que as recomendações de campanha não mudam.

---

## 7. Análise dos Resultados — Atributos Relevantes

### 7.1 Importância das features

| Ranking | Feature                  | Importância |
|---------|--------------------------|-------------|
| 1º      | `causa_acidente`         | 51,10%      |
| 2º      | `tipo_acidente`          | 43,14%      |
| 3º      | `sentido_via`            | 2,59%       |
| 4º      | `fase_dia`               | 1,06%       |
| 5º      | `uso_solo`               | 1,00%       |
| 6º      | `uf`                     | 0,46%       |
| 7º      | `tracado_via`            | 0,37%       |
| 8º      | `condicao_metereologica` | 0,18%       |
| 9º      | `dia_semana`             | 0,11%       |
| 10º     | `tipo_pista`             | 0,00%       |

### 7.2 Interpretação

**`causa_acidente` (51,10%)** é o atributo com maior poder explicativo isolado. Isso indica que a **razão pela qual o acidente ocorre** é o principal determinante de sua gravidade. Causas como falta de atenção, velocidade incompatível, ultrapassagem indevida e consumo de álcool historicamente concentram os sinistros mais graves.

**`tipo_acidente` (43,14%)** é o segundo fator mais importante. O tipo de colisão — frontal, traseira, capotamento, tombamento — está fortemente correlacionado com a existência de vítimas. Colisões frontais e capotamentos, por exemplo, geram forças de impacto muito superiores às colisões traseiras em baixa velocidade.

Juntas, essas duas features explicam **94,24%** do poder preditivo do modelo. Os demais atributos (sentido da via, fase do dia, uso do solo etc.) têm contribuição marginal, indicando que quem está envolvido no acidente e como ele acontece importa mais do que **quando** ou onde

---

## 8. Proposição de Campanhas Educativas

Com base nos resultados do modelo, propõem-se as seguintes diretrizes para campanhas e políticas públicas:

### Campanha 1 — "Causa Zero: Atenção Salva Vidas"
**Foco:** `causa_acidente` (51,10% de importância)  
**Público-alvo:** Condutores em geral, com ênfase em motoristas profissionais (caminhoneiros, motoristas de aplicativo)  
**Ações:**
- Campanhas de mídia sobre distração ao volante (uso de celular, sonolência)
- Fiscalização e blitzes educativas nas BRs com maior concentração de sinistros por falta de atenção
- Conteúdo de conscientização sobre velocidade incompatível com a via e a condição climática
- Parceria com operadoras de aplicativo para alertas in-app antes de viagens longas

### Campanha 2 — "Tipo de Colisão: o que você pode evitar"
**Foco:** `tipo_acidente` (43,14% de importância)  
**Público-alvo:** Condutores que realizam ultrapassagens e trafegam em rodovias de pista simples  
**Ações:**
- Campanhas específicas sobre riscos de colisões frontais em ultrapassagens indevidas
- Sinalização aumentada em pontos críticos de curva e declive (correlação com `tracado_via`)
- Simulações e conteúdos educativos sobre distâncias de frenagem e pontos cegos
- Reforço de fiscalização eletrônica de velocidade nos trechos com histórico de capotamentos e tombamentos

### Campanha 3 — "Noite nas Rodovias"
**Foco:** `fase_dia` + `uso_solo` (combinação)  
**Público-alvo:** Condutores em rodovias rurais no período noturno  
**Ações:**
- Alertas em aplicativos de navegação para trechos rurais no anoitecer e plena noite
- Campanhas de combate à fadiga e orientações sobre paradas obrigatórias em longas viagens
- Programa de reforço de iluminação em pontos críticos identificados pela PRF

### Priorização de recursos

| Prioridade | Ação                                      | Justificativa pelo modelo        |
|------------|-------------------------------------------|----------------------------------|
| Alta       | Campanhas sobre causas de acidentes       | Feature mais importante (51,1%)  |
| Alta       | Educação sobre tipos de colisão evitáveis | Segunda feature (43,1%)          |
| Média      | Ações noturnas em rodovias rurais         | Combinação fase_dia + uso_solo   |
| Baixa      | Campanhas por dia da semana ou clima      | Importância < 0,5% no modelo     |

---

## 9. Conclusão

O modelo de Árvore de Decisão com critério de entropia identificou que **a causa do acidente e o tipo de colisão explicam conjuntamente mais de 94% da variância preditiva** na classificação de sinistros com ou sem vítimas. Isso fornece ao gestor público um direcionamento claro: **campanhas focadas em comportamentos causadores de acidentes graves (especialmente colisões frontais e capotamentos provocados por desatenção e excesso de velocidade) têm o maior potencial de impacto na redução de vítimas nas rodovias federais brasileiras.**

O modelo apresentou acurácia de **86,75%** com alta sensibilidade para a classe "com vítimas" (recall de 100%), o que é adequado para o contexto de saúde pública, onde falsos negativos (acidentes graves classificados como leves) têm consequências mais sérias do que falsos positivos.

---

*Script de implementação disponível no repositório GitHub do projeto.*  
*Base de dados: PRF — Sinistros de Trânsito Agrupados por Ocorrência 2025 (dados.gov.br)*
