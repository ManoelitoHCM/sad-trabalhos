# =============================================================================
# TRABALHO AV1 — Sistemas de Apoio à Tomada de Decisão  (VERSÃO REVISADA)
# Disciplina: Sistemas de Apoio à Tomada de Decisão
# Curso: Sistemas de Informação
#
# Tema: Apoio à Decisão para Campanhas Educativas no Trânsito
# Algoritmo: Árvore de Decisão com critério Entropia (ID3/C4.5)
# Base de Dados: Sinistros de Trânsito PRF — 2025 (dados.gov.br)
#
# Melhorias em relação à versão inicial:
#   (1) Corrigido o bug do gráfico de importância (rótulos e cores estavam
#       invertidos em relação às barras).
#   (2) Adicionada comparação com o BASELINE (classe majoritária), para mostrar
#       o ganho real do modelo sobre o palpite trivial.
#   (3) Adicionado um modelo com class_weight='balanced', oferecendo uma leitura
#       mais honesta diante do desbalanceamento de classes.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, balanced_accuracy_score
)


# =============================================================================
# 1. CARREGAMENTO E INSPEÇÃO DOS DADOS
# =============================================================================
df = pd.read_csv('datatran2025.csv', sep=';', encoding='latin1')

print("=" * 60)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 60)
print(f"Registros carregados: {len(df):,}")
print(f"Colunas disponíveis: {df.shape[1]}")
print(f"\nDistribuição da variável 'classificacao_acidente':")
print(df['classificacao_acidente'].value_counts(dropna=False))


# =============================================================================
# 2. PREPARAÇÃO E TRATAMENTO DOS DADOS
# =============================================================================
print("\n" + "=" * 60)
print("2. PREPARAÇÃO DOS DADOS")
print("=" * 60)

df = df.dropna(subset=['classificacao_acidente'])
print(f"Registros após remoção de nulos: {len(df):,}")

# Variável alvo binária: 1 = com vítimas (feridas/fatais); 0 = sem vítimas.
# As colunas 'mortos', 'feridos', 'ilesos' são EXCLUÍDAS (são consequência do
# sinistro, não causa — incluí-las geraria vazamento de informação / data leakage).
df['com_vitimas'] = df['classificacao_acidente'].apply(
    lambda x: 0 if x == 'Sem Vítimas' else 1
)
print(f"\nVariável alvo 'com_vitimas':")
print(df['com_vitimas'].value_counts())
print(f"Proporção com vítimas: {df['com_vitimas'].mean():.1%}")

# Features de CONTEXTO (circunstâncias do sinistro, não resultados)
features = [
    'dia_semana', 'fase_dia', 'condicao_metereologica', 'tipo_pista',
    'tracado_via', 'causa_acidente', 'tipo_acidente', 'uso_solo',
    'sentido_via', 'uf'
]
X = df[features].copy()
y = df['com_vitimas']


# =============================================================================
# 3. CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# =============================================================================
# Árvores do scikit-learn exigem entrada numérica. LabelEncoder é aceitável aqui
# porque a árvore particiona por limiares, sem assumir ordem real entre categorias.
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = X[col].fillna('Ignorado').astype(str)
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le
print("\nCodificação categórica aplicada a todas as features.")


# =============================================================================
# 4. DIVISÃO TREINO / TESTE (70% / 30%, estratificada)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("\n" + "=" * 60)
print("3. DIVISÃO TREINO/TESTE")
print("=" * 60)
print(f"Treino: {len(X_train):,} registros")
print(f"Teste:  {len(X_test):,} registros")


# =============================================================================
# 5. TREINAMENTO — ÁRVORE DE DECISÃO COM ENTROPIA
# =============================================================================
print("\n" + "=" * 60)
print("4. TREINAMENTO DO MODELO")
print("=" * 60)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
clf.fit(X_train, y_train)
print("Modelo treinado com sucesso.")
print(f"Profundidade real da árvore: {clf.get_depth()}")
print(f"Número de nós folha: {clf.get_n_leaves()}")


# =============================================================================
# 6. AVALIAÇÃO DO MODELO  (+ comparação com baseline)
# =============================================================================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
cr  = classification_report(y_test, y_pred, target_names=['Sem Vítimas', 'Com Vítimas'])

# Baseline: prever sempre a classe majoritária (com vítimas)
baseline_acc = max(y_test.mean(), 1 - y_test.mean())

print("\n" + "=" * 60)
print("5. AVALIAÇÃO DO MODELO")
print("=" * 60)
print(f"\nAcurácia geral: {acc:.4f} ({acc:.1%})")
print(f"Baseline (classe majoritária): {baseline_acc:.4f} ({baseline_acc:.1%})")
print(f"Ganho do modelo sobre o baseline: {(acc - baseline_acc)*100:+.2f} p.p.")
print(f"Acurácia balanceada: {balanced_accuracy_score(y_test, y_pred):.4f}")

print("\nMatriz de Confusão:")
print("                  Previsto")
print("                  Sem Vít.  Com Vít.")
print(f"Real  Sem Vít.   {cm[0,0]:>6,}    {cm[0,1]:>6,}")
print(f"Real  Com Vít.   {cm[1,0]:>6,}    {cm[1,1]:>6,}")
print("\nRelatório Completo:")
print(cr)


# =============================================================================
# 7. MODELO ALTERNATIVO — class_weight='balanced'
# =============================================================================
# Compensa o desbalanceamento (84,6% com vítimas). Tende a reduzir a acurácia
# global, mas melhora a detecção da classe minoritária ('Sem Vítimas').
print("\n" + "=" * 60)
print("6. MODELO BALANCEADO (class_weight='balanced')")
print("=" * 60)
clf_bal = DecisionTreeClassifier(
    criterion='entropy', max_depth=6, random_state=42, class_weight='balanced'
)
clf_bal.fit(X_train, y_train)
y_bal = clf_bal.predict(X_test)
cm_bal = confusion_matrix(y_test, y_bal)
print(f"Acurácia: {accuracy_score(y_test, y_bal):.4f} | "
      f"Acurácia balanceada: {balanced_accuracy_score(y_test, y_bal):.4f}")
print(classification_report(y_test, y_bal, target_names=['Sem Vítimas', 'Com Vítimas']))


# =============================================================================
# 8. IMPORTÂNCIA DAS FEATURES (GANHO DE INFORMAÇÃO)
# =============================================================================
print("\n" + "=" * 60)
print("7. IMPORTÂNCIA DAS FEATURES")
print("=" * 60)
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:<28} {imp:.4f}  {bar}")


# =============================================================================
# 9. GRÁFICOS
# =============================================================================

# --- Gráfico 1: Importância das Features (CORRIGIDO) ---
# Ordena ASCENDENTE para o barh exibir a maior no topo sem inverter o eixo,
# evitando o desalinhamento entre barras, cores e rótulos da versão anterior.
imp_asc = importances.sort_values(ascending=True)
top2 = set(importances.head(2).index)          # duas mais importantes -> destaque
cores = ['#c0392b' if f in top2 else '#2980b9' for f in imp_asc.index]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_asc.index, imp_asc.values, color=cores)
ax.set_title('Importância das Features — Árvore de Decisão (Entropia)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Importância (Ganho de Informação)', fontsize=12)
ax.set_xlim(0, max(imp_asc.values) * 1.15)
for i, v in enumerate(imp_asc.values):          # rótulo casa com a própria barra
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("\nGráfico salvo: feature_importance.png (corrigido)")

# --- Gráfico 2: Matriz de Confusão ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=['Sem Vítimas', 'Com Vítimas']
                       ).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Matriz de Confusão', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=150)
plt.close()
print("Gráfico salvo: matriz_confusao.png")

# --- Gráfico 3: Árvore (3 primeiros níveis) ---
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(clf, max_depth=3, feature_names=features,
          class_names=['Sem Vítimas', 'Com Vítimas'],
          filled=True, rounded=True, ax=ax, fontsize=9)
ax.set_title('Árvore de Decisão — Primeiros 3 Níveis (critério: entropia)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('arvore_decisao.png', dpi=150)
plt.close()
print("Gráfico salvo: arvore_decisao.png")

print("\n" + "=" * 60)
print("Execução concluída com sucesso.")
print("=" * 60)
