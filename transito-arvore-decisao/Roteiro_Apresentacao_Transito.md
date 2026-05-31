# Roteiro de Apresentação — Apoio à Decisão no Trânsito com Árvore de Decisão
*Disciplina: Sistemas de Apoio à Tomada de Decisão. Base: PRF 2025. Duração: 6–8 min.*

---

## 1. Abertura e problema (≈45s)
Olá. Este trabalho usa aprendizado de máquina para apoiar o planejamento de
**campanhas educativas de trânsito**. O problema gerencial é direto: campanhas
costumam ser definidas por experiência e estatística geral; aqui a proposta é
orientar a decisão por dados. A pergunta central é: *quais características dos
sinistros estão mais associadas à ocorrência de vítimas e devem ser o foco
prioritário das campanhas?*

## 2. Base de dados (≈40s)
Usei a base da **Polícia Rodoviária Federal** — sinistros agrupados por ocorrência
em 2025, do portal de dados abertos. São **72.529 registros** e 30 atributos, em CSV
com separador ponto e vírgula e codificação latin-1. A variável que descreve o
resultado é a `classificacao_acidente`: sem vítimas, com vítimas feridas, ou com
vítimas fatais.

## 3. Preparação dos dados (≈1min)
Fiz duas decisões importantes na preparação. **Primeira:** transformei o alvo em
**binário** — "com vítimas" (feridas ou fatais) contra "sem vítimas" — porque o que
interessa ao gestor é distinguir o que tem consequência humana do que não tem.
**Segunda, e mais importante:** removi propositalmente as colunas `mortos`,
`feridos` e `ilesos`. Elas são **resultado** do acidente, não causa; se eu as
deixasse, o modelo acertaria de forma trivial, mas sem valor preditivo — é o que se
chama de **vazamento de informação**. Selecionei então 10 atributos de **contexto**:
causa do acidente, tipo de acidente, dia da semana, fase do dia, condição
meteorológica, tipo e traçado da via, sentido, uso do solo e UF. Como o
scikit-learn exige números, codifiquei as variáveis categóricas com LabelEncoder.

## 4. O algoritmo (≈1min)
O modelo é uma **Árvore de Decisão com critério de entropia**, que implementa o
**ganho de informação** do algoritmo ID3/C4.5. A ideia: a entropia mede a impureza
de um conjunto; a cada nó, a árvore escolhe o atributo que **mais reduz essa
incerteza** sobre a classe alvo. Limitei a profundidade a 6 níveis para equilibrar
capacidade preditiva e interpretabilidade, evitando overfitting. Dividi os dados em
70% treino e 30% teste, com estratificação para manter a proporção das classes.

## 5. Resultados e avaliação — leitura honesta (≈1min30)
*(Mostrar a matriz de confusão.)*
O modelo atingiu **86,75% de acurácia**. Mas é preciso ler esse número com cuidado:
como 84,6% dos acidentes têm vítimas, **prever sempre "com vítimas" já daria 84,6%**.
O ganho real do modelo sobre esse palpite trivial é de apenas **2 pontos**.
A matriz de confusão confirma: ele acerta praticamente todos os casos com vítimas
(recall perto de 100%), mas só identifica 14% dos casos sem vítimas. Isso é efeito
do **desbalanceamento** das classes.
*(Mostrar a tabela do modelo balanceado.)*
Para tratar isso, treinei também uma versão com **pesos balanceados**. Ela troca
acurácia global (cai para 74%) por um recall de "sem vítimas" que sobe de 14% para
**50%**, com acurácia balanceada melhor. No contexto de saúde pública, o modelo
padrão — que raramente deixa passar um caso grave — é defensável; mas mostrar as duas
versões deixa a avaliação mais transparente.

## 6. Atributos relevantes — o que decide a campanha (≈1min30)
*(Mostrar o gráfico de importância das features e a árvore.)*
Aqui está o resultado central para a decisão. Dois atributos dominam:
**a causa do acidente, com 51% de importância, e o tipo de acidente, com 43%**.
Juntos, respondem por mais de **94%** do poder preditivo. Todo o resto — dia da
semana, clima, tipo de pista — tem peso quase nulo. A leitura gerencial é clara:
**o porquê e o como do acidente importam muito mais do que o quando ou o onde.**
Importante: essas duas features lideram tanto no modelo padrão quanto no balanceado,
ou seja, a conclusão é robusta.

## 7. Proposição de campanhas (≈1min)
Com base nisso, proponho concentrar recursos em três frentes, em ordem de
prioridade:
- **Prioridade alta — causa do acidente:** campanhas contra desatenção, sonolência,
  uso de celular e velocidade incompatível, com foco em motoristas profissionais e
  blitzes educativas nos trechos críticos.
- **Prioridade alta — tipo de colisão:** campanhas sobre ultrapassagem indevida e
  colisões frontais, e fiscalização eletrônica em trechos com histórico de
  capotamento e tombamento.
- **Prioridade média — ações noturnas em rodovias rurais**, combinando fase do dia e
  uso do solo.
Campanhas genéricas por dia da semana ou clima ficam em baixa prioridade, porque os
dados mostram que têm pouco efeito sobre a gravidade.

## 8. Encerramento (≈30s)
Em resumo: a árvore de decisão transformou 72 mil registros da PRF em uma orientação
objetiva — concentrar as campanhas no **comportamento que causa o acidente** e no
**tipo de colisão evitável**, que é onde o investimento tem maior potencial de
reduzir vítimas. A principal limitação é o desbalanceamento das classes, que tratei
mostrando o baseline e o modelo balanceado. Obrigado.

---
### Checklist da gravação
- [ ] Mostrar o script rodando (preparação, treino e a saída de avaliação).
- [ ] Mostrar a matriz de confusão e citar o baseline de 84,6%.
- [ ] Mostrar a tabela comparando modelo padrão x balanceado.
- [ ] Mostrar o gráfico de importância (causa 51% + tipo 43%) e a árvore.
- [ ] Fechar com as três campanhas priorizadas pelos atributos.
