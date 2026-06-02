# Roteiro de Apresentação — Data Lake & DW de Saúde (AV2)
*Disciplina: Sistemas de Apoio à Decisão. Caso: HUOL/UFRN. Duração: 8–10 min.*

---

## 1. Abertura e contexto (≈50s)
Olá. Este trabalho apoia uma decisão de negócio real: um **hospital privado de Natal
quer se credenciar ao SUS** para ofertar internações. Antes de investir, a direção quer
entender como se comporta um hospital público de referência já credenciado — o **HUOL,
da UFRN**. Para isso, integrei **dados estruturados** (internações do SUS) e **não
estruturados** (comentários do Instagram) num **Data Lake**, montei um **Data Warehouse**
e gerei os insights que embasam a decisão de expansão.

## 2. Arquitetura da solução (≈1min)
A arquitetura segue o padrão de Data Lake em **três camadas**:
- **Bronze** — o dado cru, exatamente como veio dos 8 CSVs e dos comentários.
- **Prata** — o dado limpo e padronizado.
- **Ouro** — tabelas analíticas que respondem às perguntas de negócio.
A partir da camada ouro, derivei um **Data Warehouse em esquema estrela**. E o Data Lake
físico roda em **MinIO**, uma solução S3 gratuita, subida via Docker. Tudo é reproduzível:
um único comando executa o pipeline inteiro.

## 3. Dados estruturados — preparação (≈1min30)
*(Mostrar o script 01 e a pasta data/bronze, prata, ouro.)*
Comecei pelos dados de internação: **16 mil registros** de janeiro de 2024 a dezembro de
2025. Na limpeza, três decisões importam. **Primeira:** unifiquei as especialidades — a
base trazia 50 descrições diferentes, com variações como "oncologia clínica", "cirurgia
oncológica" e "oncologia hematologia", que reduzi a **37 especialidades** padronizadas.
**Segunda:** padronizei os nomes de município, que vinham com caixa e acentuação
inconsistentes. **Terceira:** normalizei sexo e idade, criei faixas etárias e tratei
datas e registros inválidos. O resultado é a camada prata, limpa e confiável.

## 4. Perguntas de negócio — os insights (≈2min30)
*(Mostrar os gráficos da pasta figuras, um a um.)*
Agora os achados que respondem ao que o hospital precisa saber:
- **Especialidades:** Cardiologia lidera, seguida de Urologia e Gastroenterologia —
  forte demanda cirúrgica e oncológica.
- **Perfil dos pacientes:** sexos equilibrados, mas **37% têm 60 anos ou mais** —
  demanda concentrada na terceira idade.
- **Origem:** só **35% são de Natal**; os outros 65% vêm do interior e da região
  metropolitana. O hospital atende uma área ampla, não só a capital.
- **Sazonalidade:** as internações **cresceram quase 12% de 2024 para 2025**, com picos
  no segundo semestre — útil para planejar capacidade.

## 5. Data Warehouse (≈1min)
*(Mostrar as tabelas em data/dw.)*
Modelei um Data Warehouse em **esquema estrela**: uma tabela fato com a quantidade de
internações, ligada a quatro dimensões — tempo, especialidade, município e perfil do
paciente. A soma da fato bate exatamente com o total da camada prata, o que confirma a
integridade do modelo. Com ele, o gestor consegue cruzar, por exemplo, especialidade por
faixa etária e por mês com poucos cliques.

## 6. Dados não estruturados — Instagram (≈1min30)
*(Mostrar o gráfico de sentimento e a documentação da Graph API.)*
Para as perguntas sobre a percepção do público, eu precisaria dos comentários do
Instagram do hospital. Aqui há uma limitação real e importante de explicar: a **Graph API
do Instagram só libera os comentários para quem administra a página** — o que eu, como
aluno externo, não tenho. Então fiz duas coisas: **deixei o pipeline real da API
implementado e documentado**, pronto para uso; e gerei uma **base sintética de 300
comentários** que reproduz os padrões reais. A análise — limpeza de emojis e hashtags,
tokenização e sentimento por léxico — classifica os comentários em positivo, negativo ou
neutro. O resultado: **38% positivos** (elogios à equipe), **38% negativos** (quase todos
sobre tempo de espera) e 24% neutros. A mensagem para a gestão é direta: o cuidado é bem
avaliado, mas a **espera é o ponto a corrigir**.

## 7. Data Lake físico — MinIO (≈45s)
*(Mostrar o docker-compose e o console do MinIO em http://localhost:9001.)*
Por fim, todas as camadas — bronze, prata, ouro e o Data Warehouse — são enviadas para um
bucket no **MinIO**, que é um Data Lake S3-compatível e gratuito, rodando localmente em
Docker. Um script faz o upload automático e organiza tudo por camada.

## 8. Encerramento (≈40s)
Resumindo: parti de dados públicos brutos e de comentários, construí um Data Lake em três
camadas, um Data Warehouse dimensional e uma análise que cobre as seis perguntas de
negócio. Para o hospital privado, o retrato é claro — priorizar cardiologia, urologia e
áreas cirúrgicas, preparar-se para um público idoso e de ampla origem geográfica, e tratar
o tempo de espera como diferencial competitivo. A principal limitação é o acesso real à
API do Instagram, que deixei pronto para quando houver permissão. Obrigado.

---
### Checklist da gravação
- [ ] Mostrar `run_all.py` rodando e as pastas bronze/prata/ouro sendo criadas.
- [ ] Passar pelos 4 gráficos de internação (especialidade, perfil, município, sazonalidade).
- [ ] Mostrar as tabelas do Data Warehouse (fato + dimensões).
- [ ] Mostrar o gráfico de sentimento e abrir o 02c_pipeline_graph_api.md.
- [ ] Subir o MinIO (`docker compose up -d`), rodar o upload e mostrar o console web.
