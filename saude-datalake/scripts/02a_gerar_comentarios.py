"""
02a_gerar_comentarios.py
========================
Gera a base SINTÉTICA de comentários do Instagram do HUOL (camada bronze do dado
não estruturado).

Por que sintética? A Instagram Graph API só libera os comentários de uma página
para quem a administra — o que um aluno externo não tem (ver 02c_pipeline_graph_api.md).
Para não travar o trabalho, geramos comentários que reproduzem fielmente os padrões
reais: elogios à equipe, reclamações de espera, dúvidas operacionais, além do ruído
típico (emojis, hashtags, menções) que o pipeline de limpeza (02b) precisa tratar.

O CSV de saída tem EXATAMENTE o mesmo esquema do coletor real da Graph API
(02c_pipeline_graph_api.md), de modo que 02b_analise_sentimento.py funciona sem
qualquer alteração — basta trocar a fonte quando houver acesso real.

Saída: data/bronze/comentarios_instagram_bronze.csv
Colunas: id_comentario, id_post, autor, data, texto, curtidas, rotulo_referencia

A coluna rotulo_referencia é um "gabarito" usado apenas para medir a concordância
do léxico em 02b (não é usada pela classificação em si).

Disciplina: Sistemas de Apoio à Decisão — Trabalho AV2
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(os.path.join(DATA, "bronze"), exist_ok=True)

SEED = 42
random.seed(SEED)

N_COMENTARIOS = 300

# --------------------------------------------------------------------------- #
# Bancos de frases por sentimento (contexto hospitalar real)
# --------------------------------------------------------------------------- #
POSITIVOS = [
    "Fui muito bem atendido, equipe atenciosa e competente",
    "Médicos excelentes, salvaram a vida do meu pai",
    "Atendimento humanizado, só tenho a agradecer",
    "Enfermagem nota mil, cuidaram super bem da minha mãe",
    "Hospital limpo e organizado, parabéns a todos",
    "Gratidão a toda a equipe, profissionais maravilhosos",
    "Melhor hospital público da região, recomendo demais",
    "Adorei o cuidado dos enfermeiros, muito acolhida",
    "Cirurgia foi um sucesso, equipe maravilhosa",
    "Profissionais competentes e muito humanos, obrigado",
    "Fui acolhido com muito carinho, atendimento excelente",
    "Estrutura ótima e médicos atenciosos, amei",
    "Tratamento oncológico de excelência, sou grato",
    "Equipe de plantão maravilhosa, me senti seguro",
    "Atendimento rápido e eficiente dessa vez, parabéns",
]
NEGATIVOS = [
    "Esperei horas para ser atendido, um absurdo",
    "Demora enorme na fila, péssimo atendimento",
    "Fiquei no corredor por falta de leito, lamentável",
    "Consulta remarcada de novo, descaso com o paciente",
    "Atendimento demorado e poucos médicos de plantão",
    "Esperei a manhã toda, fila gigante e ninguém explica nada",
    "Banheiro sujo e estrutura precária, revoltante",
    "Faltou medicamento e ninguém resolveu, triste",
    "Demora absurda para marcar exame, péssimo",
    "Espera de horas no pronto socorro, descaso total",
    "Equipamento quebrado e atendimento ruim, lamentável",
    "Marquei e foi remarcada sem aviso, um problema sério",
    "Muita demora e pouca informação, atendimento falho",
    "Fila enorme desde cedo, esperei demais para nada",
]
NEUTROS = [
    "Alguém sabe o horário de visita?",
    "Qual o telefone para marcar consulta?",
    "O hospital atende pelo convênio ou só SUS?",
    "Onde fica a entrada da emergência?",
    "Precisa de encaminhamento para a cardiologia?",
    "Vocês fazem cirurgia bariátrica pelo SUS?",
    "Como faço para conseguir a segunda via do laudo?",
    "Tem estacionamento para acompanhante?",
    "Qual documento preciso levar para internação?",
    "A ala de oncologia funciona aos fins de semana?",
    "Quero saber sobre o agendamento de exames",
    "Bom dia, o ambulatório abre que horas?",
]

# Ruído típico de redes sociais para o pipeline de limpeza tratar
EMOJIS = ["😡", "❤️", "🙏", "👏", "😢", "🏥", "👍", "😍", "🤬", "✨", "💉", "😊"]
HASHTAGS = ["#SUS", "#HUOL", "#saude", "#Natal", "#UFRN", "#hospitalpublico", "#gratidao"]
MENCOES = ["@huol_ufrn", "@ebserh", "@meuamigo", "@familia"]
AUTORES = [
    "maria.silva", "joao_santos", "ana.paula87", "carlos.rn", "luana_oli",
    "pedro.almeida", "fernanda.lima", "rafael_sz", "julia.costa", "marcos.vf",
    "patricia_n", "thiago.rn", "beatriz.alves", "gustavo_p", "sandra.maria",
]


def _enfeitar(texto: str) -> str:
    """Adiciona ruído realista (emojis, hashtags, menções) a parte dos textos."""
    if random.random() < 0.45:
        texto += " " + random.choice(EMOJIS)
    if random.random() < 0.30:
        texto += " " + random.choice(EMOJIS)
    if random.random() < 0.35:
        texto += " " + random.choice(HASHTAGS)
    if random.random() < 0.15:
        texto = random.choice(MENCOES) + " " + texto
    return texto


def gerar():
    # Mix proporcional próximo de um cenário real: positivo dominante, forte
    # presença de reclamações de espera, e uma fração de dúvidas (neutro).
    bancos = (
        [("positivo", POSITIVOS)] * 43 +
        [("negativo", NEGATIVOS)] * 39 +
        [("neutro", NEUTROS)] * 18
    )

    data_base = datetime(2025, 1, 1)
    posts = [f"post_{i:03d}" for i in range(1, 21)]

    linhas = []
    for i in range(1, N_COMENTARIOS + 1):
        rotulo, banco = random.choice(bancos)
        texto = _enfeitar(random.choice(banco))
        dias = random.randint(0, 360)
        minutos = random.randint(0, 1439)
        data = data_base + timedelta(days=dias, minutes=minutos)
        linhas.append({
            "id_comentario": f"c_{i:04d}",
            "id_post": random.choice(posts),
            "autor": random.choice(AUTORES),
            "data": data.strftime("%Y-%m-%dT%H:%M:%S+0000"),
            "texto": texto,
            "curtidas": max(0, int(random.gauss(4, 6))),
            "rotulo_referencia": rotulo,
        })

    df = pd.DataFrame(linhas)
    destino = os.path.join(DATA, "bronze", "comentarios_instagram_bronze.csv")
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"[BRONZE] {len(df)} comentários sintéticos gerados -> {destino}")
    print("Distribuição de referência (gabarito):")
    print(df["rotulo_referencia"].value_counts().to_string())
    return df


if __name__ == "__main__":
    gerar()
