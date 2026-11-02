# Diário do projeto: quem cai no Brasileirão 2026?

O registro completo de cada decisão, teste e resultado, na ordem em que aconteceram. O resumo do projeto está no [README](README.md).

## Ideia geral

1. **Modelo de partidas** – treinado com milhares de jogos desde 2012, estima a chance de vitória do mandante, empate e vitória do visitante em cada jogo.
2. **Simulação Monte Carlo** – sorteia os jogos que faltam de acordo com essas probabilidades, milhares de vezes, e conta em quantas simulações cada time termina entre os 4 últimos.
3. **Backtest** – repete o processo em temporadas passadas, "congeladas" no mesmo ponto do campeonato, para medir se as probabilidades são confiáveis.

## Metodologia: três modelos, a mesma régua

| Etapa | Modelo | O que ele traz | Papel |
|---|---|---|---|
| 1 | **Regressão logística multinomial** (Elo + features) | Simples, interpretável, probabilidades bem calibradas | Modelo base: o ponto de comparação |
| 2 | **XGBoost** (mesmas features) | Captura relações não lineares e interações entre features | Testa se um modelo mais flexível supera o base |
| 3 | **Modelo hierárquico bayesiano** (Poisson de ataque/defesa, Baio & Blangiardo 2010, em PyMC) | Força de cada time com **incerteza**, atualizada rodada a rodada; a priori de cada temporada é a força anterior puxada para a média | Testa se levar a incerteza para a simulação dá probabilidades mais honestas |

**Ordem:** 1 → 2 → 3. O XGBoost reaproveita as features e o código da etapa 1, então vem logo depois. O bayesiano é um caminho diferente (modela gols, não resultados) e fica por último.

**Como os três serão comparados**
- **Por partida:** log loss e Brier score nas probabilidades de vitória, empate e derrota, mais um gráfico de calibração. As odds de mercado servem de referência ("quão perto do mercado chegamos?").
- **Por rebaixamento:** backtest nas temporadas 2015–2025, congeladas em várias rodadas (ex.: 20, 25, 30). Medimos o Brier score das probabilidades de rebaixamento e a calibração.
- **Validação temporal:** cada modelo só treina com jogos anteriores ao período avaliado, sem embaralhar os dados, para não vazar informação do futuro. Isso é especialmente importante para o XGBoost, que tem mais facilidade de decorar os dados.

Os três modelos se atualizam a cada rodada: basta rodar os scripts de novo com os resultados novos.

## Plano de features

Critérios para uma feature entrar: (1) existir também para os jogos futuros, (2) trazer informação que o Elo ainda não tem, (3) ter um motivo plausível. Cada feature entra por **teste incremental**: só fica se melhorar o erro em temporadas que o modelo não viu.

| Prioridade | Feature | Hipótese | Status |
|---|---|---|---|
| 1 | Diferença de Elo (versão que considera o placar) | O mais forte vence mais; goleadas dizem mais que vitórias magras | **aprovada** (Elo com placar venceu o simples) |
| 1 | Rodagem: temporadas na Série A nas últimas 10 | Promovidos com pouca experiência na elite sofrem mais | testada: não acrescentou ao Elo |
| 1 | Pontos, gols marcados e gols sofridos nos últimos 5 jogos | Momento importa; sofrer muitos gols é decisivo na luta contra a queda | testada: não acrescentou ao Elo |
| 1 | Elo do mercado (nota corrigida pelas odds dos jogos passados) | O mercado sabe coisas que os placares não mostram | **aprovada**: virou o modelo final |
| 2 | Histórico do confronto direto ("freguesia"), resultado do 1º turno e clássico estadual | Certos confrontos têm padrão próprio além da força dos times | testada sobre o Elo do mercado: nenhuma acrescentou (h2h +0,0006 ± 0,0003; ida/volta +0,0005 ± 0,0006; clássico +0,0004 ± 0,0003). Correlação entre a "surpresa" do 1º turno e a do returno: 0,01 em 2.778 pares |
| 2 | Motivação: "na briga" e "desenganado" na reta final | Quem briga contra a queda se esforça; quem já não tem chance relaxa | testada: efeito na direção esperada, mas pequeno demais para melhorar o modelo |
| 2 | Goleada no último jogo | Efeito de confiança, além da força | a fazer |
| 2 | Jogos desde a troca de técnico | Efeito "técnico novo", controlando pela regressão à média | a fazer (técnicos 2014–2024 disponíveis; 2025–26 a coletar) |
| 3 | Finanças (dívida e receita dos balanços anuais) | Crise financeira afeta o desempenho | fase posterior |
| – | Dias de descanso | Cansaço | descartada: os dados não têm os jogos de copas, e a hipótese parece fraca |

**Simulação dinâmica:** dentro de cada simulação, Elo, forma e folga são atualizados a cada jogo sorteado, na ordem real do calendário (`data/manual/calendario_2026.csv`).

## Estrutura

```
atualizar.py              atualização completa de cada rodada (chama os scripts abaixo)
01_coleta_dados.py        coleta, limpeza e checagens dos dados
02_features.py            Elo, Elo do mercado, forma recente e outras features
03_modelo_logistico.py    etapa 1: regressão logística (modelo principal)
04_modelo_xgboost.py      etapa 2: XGBoost (não adotado)
05_modelo_bayesiano.py    etapa 3: bayesiano com simulação e backtest próprios (segunda opinião)
06_simulacao.py           Monte Carlo das rodadas restantes com o modelo principal
07_backtest.py            validação em temporadas passadas
pagina/                   modelo da página; o atualizar.py coloca os números nele
apresentacao.html         página gerada (não editar à mão: é sobrescrita a cada rodada)
scripts/                  notebooks de prática
data/raw/                 arquivos baixados, sem alteração
data/manual/              calendário, resultados recentes digitados, valor dos elencos
data/processed/           arquivos gerados pelos scripts (inclui historico_previsoes.csv)
logs/                     registro de cada atualização
```

## Como rodar

```bash
pip install pandas matplotlib scikit-learn ipykernel optuna pymc nutpie xgboost
python 01_coleta_dados.py                                 # processa os dados
python 01_coleta_dados.py --forcar-download --conferir    # baixa de novo e compara com uma 2ª fonte
python atualizar.py                                       # atualização completa da rodada (~5 min)
python atualizar.py --agendar 09:00                       # agenda a atualização diária no Windows
```

## Atualização automática (`atualizar.py`)

Um comando faz a rodada inteira: baixa os dados, confere, recalcula as features, simula com os dois modelos, guarda o histórico e gera a página.

1. **Download** do `BRA.csv` (`01`). Para se o football-data trouxer um nome de time desconhecido.
2. **Conferência** dos jogos de 2026 (football-data + `data/manual/resultados_recentes_2026.csv`): 20 times, nenhum jogo repetido, ninguém com mais de 38 jogos, conhecidos + a jogar = 380, e nenhum placar manual que contradiga o football-data. Se algo falhar, nada é publicado e o registro explica o motivo.
3. **Nada novo?** Se o número de jogos, de jogos com odds e de gols for o mesmo da última vez, encerra em segundos. Por isso dá para agendar todo dia.
4. **Modelos:** `02` e `06` (principal) e `05` (bayesiano). Na primeira execução, ou se o modelo salvo for de antes do Elo do mercado, também roda `02 --otimizar` e `03`. Se o bayesiano falhar, a página sai só com o principal, com um aviso.
5. **Saídas conferidas** (20 times, probabilidades de queda somando 4), **histórico** em `data/processed/historico_previsoes.csv` (uma linha por time e rodada) e **página** gerada a partir de `pagina/modelo_apresentacao.html`. A página continua um arquivo único: abre com dois cliques e vai servir também para o GitHub Pages.

Cada execução grava `logs/atualizacao_AAAA-MM-DD_HHMM.log`. `--agendar HH:MM` cria uma tarefa diária no Agendador de Tarefas do Windows usando o mesmo Python do comando (o que tem PyMC). Para conferir, procure "Brasileirao - atualizar previsao" no Agendador: "Último resultado" 0x0 = deu certo, 0x1 = alguma conferência falhou.

**Resultado manual:** se a rodada acabou e o football-data ainda não publicou, digite os placares em `resultados_recentes_2026.csv` e rode `python atualizar.py --sem-download`. Esses jogos entram na simulação, mas sem odds. Quando o football-data os publicar, a atualização seguinte percebe as odds novas e roda de novo.

## Fontes de dados

- [football-data.co.uk](https://www.football-data.co.uk/) – resultados e odds da Série A desde 2012 (fonte principal).
- [adaoduque/Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset) – resultados de 2003 a 2024 (usado para conferência).

## Decisões registradas

| Decisão | Motivo |
|---|---|
| Prever partidas em vez de "cai / não cai" direto | Só ~56 rebaixamentos desde 2012: pouco para treinar; partidas são ~5 mil exemplos e capturam a tabela restante |
| Odds do próprio jogo como **referência**, não como feature | Para os jogos que faltam em 2026 ainda não há odds |
| Modelo final usa o **Elo do mercado** (odds de jogos passados) | As odds passadas viram uma nota de força que vale para os jogos futuros; melhorou o log loss em 10 de 10 temporadas e o Brier de rebaixamento em 9% |
| Usar odds de fechamento da Pinnacle quando houver | São a referência mais precisa do mercado |
| Probabilidades das odds pelo método proporcional | Simples para começar; métodos mais refinados (Shin, power) podem ser testados depois |
| Punição do STJD em 2013 aplicada manualmente | A soma de pontos jogo a jogo não reflete a decisão do tribunal |
| Em 2026 as odds vêm da média do mercado, e não da Pinnacle | O arquivo não traz odds de fechamento da Pinnacle a partir de meados de 2025 |
| Conferência entre fontes feita por temporada + confronto | O football-data usa o horário do Reino Unido, então jogos noturnos aparecem com a data do dia seguinte |
| Fator casa estimado com mais peso nas temporadas recentes | Prática 1: vitórias do mandante caíram de 50,9% (2012–16) para 46,9% (2021–25), com ~1.900 jogos em cada período. Uma temporada isolada oscila ±5 p.p. só por acaso, então comparamos blocos de anos |
| Fator casa por time fica para depois | Cada time joga só 19 vezes em casa por ano: estimativa muito ruidosa |
| Três modelos (logística, XGBoost, bayesiano) com a mesma avaliação | Permite responder se complexidade extra ou incerteza explícita melhoram as probabilidades de rebaixamento |
| Calendário restante digitado em `data/manual/calendario_2026.csv` | O football-data não traz jogos futuros. Fonte: páginas de rodada do ValorFinal, conferidas contra os 71 confrontos calculados pelo script 01 (bateram 100%) |
| Chapecoense x Vasco (rodada 21, adiado e sem data) simulado no fim | Sem data definida; a posição dele na ordem pouco altera o resultado |
| Resultados recentes digitados em `data/manual/resultados_recentes_2026.csv` | O football-data demora alguns dias para atualizar; a simulação usa esses placares fixos e os ignora quando eles chegam ao arquivo oficial |
| Placar da simulação sorteado a partir dos placares reais, dado o resultado | O modelo só prevê M/E/V, mas os desempates e o Elo com placar precisam de gols |
| Incerteza no Elo inicial (`INCERTEZA_PADRAO`): era 75, agora 0 | Com o Elo comum, 75 corrigia o excesso de confiança. Com o Elo do mercado, o modelo já sai calibrado e a incerteza extra piora o Brier |
| Forma recente calculada só dentro da temporada | Jogos do ano anterior refletem outro elenco; no 1º jogo da temporada usamos valores neutros e a coluna `jogos_janela` avisa o modelo |

## Resultados

### Etapa 1 – Regressão logística (`03_modelo_logistico.py`)

Validação walk-forward: para cada temporada de 2017 a 2026, treina com as anteriores e prevê a temporada inteira. Log loss (menor = melhor):

| Modelo | Log loss | Diferença vs. base |
|---|---|---|
| Chute pelas frequências históricas | 1,0588 | |
| **Base: só diferença de Elo** | **1,0237** | |
| Base + rodagem | 1,0237 | +0,0000 ± 0,0014 |
| Base + pontos nos últimos 5 | 1,0240 | +0,0004 ± 0,0003 |
| Base + saldo nos últimos 5 | 1,0240 | +0,0003 ± 0,0004 |
| Base + gols sofridos nos últimos 5 | 1,0235 | −0,0001 ± 0,0002 |
| Base + todas | 1,0250 | +0,0013 ± 0,0017 |
| Odds do mercado | 1,0001 | |

**Conclusões**
- Nenhuma feature de prioridade 1 acrescentou informação ao Elo. O Elo já é construído com os mesmos resultados: o "momento" já entra nele pelos ajustes jogo a jogo, e a rodagem pela nota inicial dos promovidos.
- O mercado continua bem à frente. A diferença deve vir de informação que não está nos placares: escalações, lesões, bastidores. É aí que features externas (técnico, finanças) podem fazer diferença.
- O modelo superestima a vitória do mandante em ~2,4 p.p. (previsto 49,6%, real 47,2%), coerente com a queda do fator casa. O peso por recência reduz esse viés, mas não melhora o log loss de forma detectável. **Decisão:** adotar meia-vida de 6 temporadas no modelo final (`MEIA_VIDA_FINAL`), porque três evidências independentes apontam a queda do fator casa e o custo é praticamente zero.
- Teste ao vivo: a rodada 32 não estava nos dados. O modelo deu 22% à vitória da Chapecoense sobre o Fluminense (aconteceu) e 70% à do Cruzeiro sobre o Remo (aconteceu).

### Simulação Monte Carlo com a etapa 1 (`06_simulacao.py`), após a rodada 32

20.000 simulações, Elo atualizado a cada jogo sorteado, placares sorteados a partir dos placares reais de 2021–2025.

| Time | Pontos hoje | Jogos restantes | Pontos finais (média) | Rebaixamento | ValorFinal (comparação) |
|---|---|---|---|---|---|
| Chapecoense | 21 | 7 | 26,8 | 100,0% | 99,9% |
| Remo | 26 | 6 | 32,2 | 99,8% | 99,5% |
| Grêmio | 34 | 6 | 40,7 | 61,4% | 60,9% |
| Corinthians | 33 | 6 | 40,7 | 57,2% | 58,7% |
| Internacional | 35 | 6 | 42,8 | 33,7% | 34,8% |
| Mirassol | 37 | 6 | 44,9 | 16,1% | 16,9% |
| Coritiba | 39 | 6 | 44,8 | 12,6% | 9,2% |
| Vitória | 38 | 6 | 45,3 | 10,4% | 9,4% |
| Vasco | 38 | 7 | 46,2 | 6,4% | 4,7% |
| Botafogo | 39 | 6 | 47,8 | 2,3% | 5,6% |

- **Pontos para se salvar:** o 17º colocado termina com 41 pontos na mediana (entre 39 e 43 em 80% das simulações). Com 43 pontos, um time se salva em ~90% dos cenários.
- **Quarteto mais provável:** Chapecoense, Corinthians, Grêmio e Remo (27%).
- **Elo dinâmico x congelado:** praticamente iguais com 6 rodadas restantes (diferenças ≤ 1,4 p.p.). A atualização dinâmica deve pesar mais em simulações feitas no meio do campeonato, o que o backtest vai mostrar.
- A comparação com o [ValorFinal](https://valorfinal.com.br/brasileirao/chances-rebaixamento), um modelo independente, mostra diferenças de no máximo 3,4 p.p.

### Backtest (`07_backtest.py`)

Temporadas 2015–2025, campeonato "congelado" nas rodadas 10, 15, 20, 25, 30 e 34, simulando o resto só com informação disponível naquele momento (5.000 simulações por cenário). Brier das probabilidades de rebaixamento (menor = melhor; chutar 20% para todos dá 0,160):

| Corte | Modelo (Elo dinâmico) | Modelo (Elo congelado) | Histórico por posição | Z4 atual |
|---|---|---|---|---|
| Rodada 10 | **0,123** | 0,126 | 0,146 | 0,191 |
| Rodada 15 | **0,109** | 0,112 | 0,131 | 0,209 |
| Rodada 20 | **0,086** | 0,087 | 0,092 | 0,127 |
| Rodada 25 | **0,068** | 0,068 | 0,069 | 0,091 |
| Rodada 30 | **0,058** | 0,058 | 0,066 | 0,100 |
| Rodada 34 | 0,048 | 0,048 | **0,048** | 0,055 |
| Todos | **0,082** | 0,083 | 0,092 | 0,129 |

No período limpo (2022–2025, que não influenciou os parâmetros do Elo), a ordem se mantém no total: 0,098 (modelo) contra 0,106 (histórico) e 0,154 (Z4 atual).

**Calibração (todos os cenários)**

| Faixa prevista | Previsões | Prob. média | Frequência real |
|---|---|---|---|
| 0–2% | 596 | 0,2% | 0,7% |
| 2–10% | 198 | 5,2% | 7,1% |
| 10–25% | 150 | 17,3% | 19,3% |
| 25–50% | 161 | 36,7% | 35,4% |
| 50–75% | 97 | 61,7% | 54,6% |
| 75–90% | 47 | 81,5% | 76,6% |
| 90–98% | 27 | 94,4% | 100% |
| 98–100% | 44 | 99,5% | 100% |

**Conclusões**
- O modelo supera as duas linhas de base em quase todos os cortes; a vantagem é maior no início do campeonato, quando a tabela ainda engana.
- O Elo dinâmico é melhor que o congelado nas rodadas 10–15 e igual no fim, como esperado.
- **Leve excesso de confiança na faixa intermediária:** quem recebe 50–90% cai um pouco menos do que o previsto, e quem recebe 2–25% cai um pouco mais. Na rodada 34, o simples "histórico por posição" empata com o modelo. As salvações improváveis (Juventude 2021, Vasco 2023, Vitória 2024 e 2025) sugerem que times ameaçados reagem no fim, o que é exatamente a hipótese de **motivação** (prioridade 2), e também o tipo de incerteza que a **etapa bayesiana** pretende capturar. Com ~100 previsões por faixa, o efeito ainda está perto do limite do acaso.
- Maiores surpresas na rodada 30: Bahia 2021 (11,7%), Ceará 2025 (13,8%), Athletico-PR 2024 (13,9%), Criciúma 2024 (18,9%) e Cruzeiro 2019 (21,1%).

### Motivação e incerteza (tentativas de corrigir o excesso de confiança)

**Análise exploratória** – pontos reais menos pontos esperados pelo modelo, por jogo, nos últimos 10 jogos de cada time (2017–2025):

| Situação antes do jogo | Jogos | Diferença por jogo |
|---|---|---|
| Desenganado (abaixo da salvação por mais de 1,5 ponto por jogo restante) | 139 | −0,20 (z = −2,1) |
| No Z4 | 277 | +0,09 |
| Até 3 pontos acima do Z4 | 211 | +0,05 |
| Ameaçado (folga ≤ jogos restantes) | 214 | +0,09 |
| Seguro | 1.017 | −0,02 |
| Controle: 1º turno, mesmas categorias | | ≈ 0 |

A direção confirma a hipótese (quem briga contra a queda rende um pouco mais; quem já está sem chance rende menos), mas o efeito é pequeno.

**Como feature** (`na_briga`, `desenganado`, calculadas em `02_features.py` e dentro de cada simulação): não passou na regra no modelo de jogos (−0,0001 ± 0,0006) e quase não mudou o backtest (Brier 0,0818 contra 0,0820). Não foi adotada.

**Incerteza sobre a força dos times** (`--incerteza-elo`): cada simulação soma ao Elo inicial um sorteio normal com desvio padrão σ.

| Configuração | Brier total | Brier 2022–25 | Faixa 50–90%: prevista / real | Faixa 2–25%: prevista / real |
|---|---|---|---|---|
| Atual (σ = 0) | 0,0820 | 0,0980 | 68% / 62% | 10,4% / 12,4% |
| Motivação | 0,0818 | 0,0975 | 68% / 61% | 10,3% / 12,0% |
| σ = 30 | 0,0819 | 0,0973 | 67% / 63% | 10,4% / 11,7% |
| σ = 50 | 0,0818 | 0,0963 | 67% / 65% | 10,4% / 10,1% |
| **σ = 75** | **0,0817** | **0,0949** | **67% / 68%** | **10,5% / 9,6%** |
| σ = 100 | 0,0822 | 0,0941 | 67% / 73% | 10,9% / 9,1% |

Com σ = 75 o excesso de confiança desaparece; com σ = 100 o modelo passa a ser inseguro demais. O Brier total quase não muda (as diferenças estão dentro do ruído), mas a calibração melhora claramente, e é ela que importa para ler "58% de chance" ao pé da letra. Ressalva: σ foi escolhido olhando o próprio backtest, então o ganho está um pouco otimista. A etapa bayesiana deve estimar essa incerteza diretamente dos dados.

### Etapa 2 – XGBoost (`04_modelo_xgboost.py`)

Hiperparâmetros escolhidos pelo Optuna (100 tentativas) usando **só 2017–2021**; comparação final em **2022–2026**, temporadas que a otimização nunca viu. Mesmo peso por recência da etapa 1.

| Modelo (teste 2022–2026) | Log loss | vs. logística |
|---|---|---|
| Regressão logística, só Elo (etapa 1) | **1,0214** | |
| XGBoost, só Elo | 1,0236 | +0,0022 ± 0,0015 (melhor em 2 de 5 temporadas) |
| XGBoost, todas as 19 features | 1,0219 | +0,0005 ± 0,0019 (melhor em 2 de 5 temporadas) |
| Odds do mercado | 0,9986 | |

**Conclusões**
- O XGBoost **não superou** a regressão logística, nem com o Elo sozinho, nem com todas as features.
- O sinal mais forte: nas duas versões, o Optuna escolheu **árvores de profundidade 1** (uma única pergunta por árvore). Ou seja, o melhor XGBoost possível foi o que **não usa interações** entre features, quase um modelo linear. Os dados estão dizendo que não há estrutura complexa a explorar.
- A "importância das features" mostra rodagem, forma e folga sendo usadas pelo modelo, mas isso não melhorou as previsões fora da amostra. Importância mede o que o modelo usou, não o que ajudou.
- Como não passou na regra, o XGBoost não foi levado para a simulação nem para o backtest de rebaixamento.

### Etapa 3 – Modelo hierárquico bayesiano (`05_modelo_bayesiano.py`)

Poisson de ataque e defesa (Baio & Blangiardo 2010), com forças que mudam entre temporadas por um processo AR(1) e efeito para promovidos. Janela de 4 temporadas, ajustado em PyMC com o amostrador nutpie (4 cadeias × 1.000 sorteios).

**O que o modelo aprendeu (2023–2026)**

| Parâmetro | Média | Intervalo de 89% | Leitura |
|---|---|---|---|
| Persistência | 0,87 | 0,74–0,96 | 87% da força passa de um ano para o outro (a regressão à média, agora estimada) |
| Mudança anual | 0,08 | 0,05–0,12 | Quanto a força muda tipicamente entre temporadas |
| Casa | 0,33 | 0,27–0,38 | Mandante marca ~39% mais gols |
| Promovido (ataque / defesa) | −0,02 / +0,03 | ambos incluem 0 | Sem efeito claro além da força estimada |

**2026 (após a rodada 32):** Grêmio 59,4%, Corinthians 59,2%, Internacional 32,4%, Mirassol 15,3%, Coritiba 14,0%, Vitória 11,1%, Vasco 6,4% (etapa 1: 58,6%, 53,6%, 33,7%, 17,9%, 13,2%, 11,7%, 7,9%).

**Backtest** (mesmos cenários da etapa 1, cortes nas rodadas 10, 20, 30 e 34):

| Corte | Bayesiano | Etapa 1 (σ = 75) | Histórico por posição |
|---|---|---|---|
| Rodada 10 | 0,127 | **0,122** | 0,146 |
| Rodada 20 | **0,086** | 0,088 | 0,092 |
| Rodada 30 | **0,057** | 0,058 | 0,066 |
| Rodada 34 | 0,048 | 0,048 | **0,048** |
| Todos | 0,080 | **0,079** | 0,088 |
| Todos, 2022–25 | 0,093 | **0,092** | 0,102 |

**Conclusões**
- **Empate técnico** com a etapa 1: diferença de +0,0007 ± 0,0013 no Brier. A média dos dois modelos também empata (0,0788).
- Mas o bayesiano chega lá **sem nenhum ajuste manual**: a etapa 1 precisou escolher σ = 75 olhando o próprio backtest, enquanto o bayesiano estima a incerteza dos dados. É a forma mais honesta de chegar ao mesmo resultado.
- Calibração: na faixa de 50–90%, previu 67,5% e aconteceram 64,4%. É melhor que a etapa 1 sem incerteza (67,6% / 63,2%), mas um pouco menos calibrado que a etapa 1 com σ = 75.
- Pior no início do campeonato (rodada 10): as forças são constantes dentro da temporada, e os gols de um único ano mudam pouco a estimativa.
- Diagnósticos: 42 dos 44 ajustes com r_hat ≤ 1,01; os outros 2 (2021) com 1,011, aceitável. Nenhuma divergência.

**Ideias para melhorar o bayesiano:** correção de Dixon-Coles para empates; forças que mudam também dentro da temporada; usar o valor de mercado do elenco como priori. (Os números acima são contra a etapa 1 antiga, com Elo comum. A força dentro da temporada, mais a fusão com as odds, foi testada na seção "Bayesiano 2.0", mais abaixo. O Dixon-Coles perdeu prioridade: em 2025, a Poisson previu 25,6% de empates, e aconteceram 26,1%.)

### Melhoria: Elo do mercado (modelo atual)

Um Elo em que, depois de cada jogo, a nota é corrigida em direção ao que o **mercado esperava** antes do jogo (P(vitória) + 0,5 × P(empate), tirado das odds), e não ao resultado. A nota absorve o que as casas de apostas sabem (lesões, escalações, crises) e continua valendo para jogos futuros, que não têm odds. Sem vazamento: as odds são conhecidas antes do jogo. Hiperparâmetros escolhidos pelo Optuna com 2014–2021 (K ≈ 146, regressão ≈ 0,57, promovido ≈ 1355). Na simulação, depois do ponto de partida, a nota passa a ser corrigida pelos resultados sorteados, com o K do Elo comum.

**Jogo a jogo** (walk-forward, log loss):

| Modelo | 2017–2026 | 2022–2026 |
|---|---|---|
| Elo comum | 1,0238 | 1,0214 |
| **Elo do mercado** | **1,0141** | **1,0151** |
| Elo comum + Elo do mercado | 1,0148 | |
| Odds do próprio jogo (referência) | 1,0001 | 0,9986 |

Melhora de −0,0098 ± 0,0024, melhor em **10 de 10 temporadas**: a primeira ideia a passar com folga na regra. Fecha ~40% da distância para o mercado. Somar o Elo comum não ajuda: a nota do mercado já contém o que os resultados ensinam.

**Rebaixamento** (backtest, Brier):

| Configuração | Total | 2022–25 | Rodada 10 | Rodada 30 | Faixa 50–90%: prevista / real |
|---|---|---|---|---|---|
| Elo comum, σ = 75 (anterior) | 0,0817 | 0,0949 | 0,122 | 0,058 | 67% / 68% |
| **Elo do mercado, σ = 0** | **0,0746** | **0,0852** | **0,104** | **0,055** | 69% / 67% |
| Elo do mercado, σ = 75 | 0,0759 | 0,0855 | 0,107 | 0,056 | 68% / 75% |
| Histórico por posição | 0,0919 | 0,1055 | 0,146 | 0,066 | |

Diferença contra o modelo anterior: −0,0072 ± 0,0016. O erro ficou **19% menor** que o do "histórico por posição" (antes, 11%). A maior melhora é no início do campeonato. E a incerteza manual deixou de ser necessária: com σ = 0 o modelo já sai calibrado, e qualquer incerteza extra o deixa inseguro demais. Isso sugere que o excesso de confiança anterior vinha de o Elo comum estimar mal a força dos times. Nos mesmos 4 cortes do backtest bayesiano, o Elo do mercado tem Brier 0,072 contra 0,080 do bayesiano.

**2026 (após a rodada 32):** Grêmio 74,4%, Corinthians 33,8%, Internacional 29,4%, Coritiba 21,6%, Mirassol 18,1%, Vitória 16,8%, Botafogo 4,4%, Vasco 1,5%. A maior mudança em relação ao Elo comum: o mercado considera Corinthians e Vasco bem mais fortes, e Grêmio e Coritiba mais fracos, do que os resultados sugerem.

### Tentativas de melhorar o Elo do mercado (sem sucesso)

| Ideia | Resultado (sobre o Elo do mercado) |
|---|---|
| Termo para empates (\|diferença de Elo\|) | −0,0002 ± 0,0008 no log loss: ruído |
| Peso por recência (meia-vida 3, 10 ou nenhuma) | diferenças ≤ 0,0003: ruído |
| K maior nos primeiros jogos da temporada (×1,5 ou ×2) | diferenças ≤ 0,0002: ruído |
| K da nota do mercado na simulação (0 a 40) | Brier em U, mínimo no valor atual (K ≈ 12, o mesmo do Elo comum) |
| Histórico do confronto direto, 1º turno, clássico estadual | ver plano de features: nenhum acrescentou |

### Valor de mercado do elenco (testado, não adotado)

Valores do início de cada temporada, 2019–2026, em `data/manual/valor_elenco.csv` (fontes por temporada na coluna `observacao`; 2020–2022 enviados pelo José). Usado como log do valor relativo à média da temporada, o que elimina o efeito da moeda (R$ ou €) e da inflação. Correlação com o Elo do mercado no 1º jogo de cada temporada: entre 0,47 (2022) e 0,88 (2020).

| Uso (walk-forward 2021–2026, sobre o Elo do mercado) | Log loss, todos os jogos | Só os 10 primeiros jogos |
|---|---|---|
| Feature direta (diferença de valor) | +0,0015 ± 0,0013 | +0,0029 ± 0,0026 |
| Feature só no início da temporada | +0,0016 ± 0,0011 | +0,0066 ± 0,0039 |
| Nota de partida do Elo do mercado (β = 50 / 100 / 150) | +0,0010 / +0,0034 / +0,0067 | +0,003 / +0,010 / +0,020 |
| Nota de partida só para promovidos (β = 100 / 200) | +0,0025 / +0,0064 | +0,008 / +0,021 |

Todas as versões pioraram, e quanto mais peso o valor recebe, pior. Conclusão: **as odds já incorporam o valor dos elencos desde o primeiro jogo**, e o Elo do mercado, com K alto, absorve isso em poucas rodadas. Casos como o Botafogo de 2022 (SAF) e o Santos de 2025 (Neymar), em que o valor destoava do Elo inicial, são corrigidos pelo mercado logo nas primeiras odds.

Com os dados disponíveis, o modelo parece ter chegado ao teto. Os próximos ganhos dependem de **informação nova**. O xG, que seria a melhor candidata, não está disponível em fonte aberta: o FBref não traz mais xG do Brasileirão (a coluna existe, mas vem zerada).

### Bayesiano 2.0: fusão com o mercado e força dentro da temporada

Duas mudanças no `05_modelo_bayesiano.py`, agora ligadas por padrão:

1. **Fusão com o mercado:** a mesma força de cada time (ataque − defesa) precisa explicar os gols **e** o que o mercado esperava antes de cada jogo já disputado: logit(P(vitória) + 0,5 × P(empate)) ~ Normal(h + κ × (força do mandante − força do visitante), σ). O modelo estima sozinho quanto confiar nas odds (σ ≈ 0,21, bem mais preciso que os gols). Os jogos futuros continuam sem odds: são simulados só com as forças.
2. **Força em 4 blocos por temporada:** de um bloco de ~95 jogos para o outro, a força dá um passo aleatório de tamanho estimado.

**Backtest 2015–2025** (mesmos 44 cenários; 4 cadeias × 500 sorteios):

| Versão | Rodada 10 | Rodada 20 | Rodada 30 | Rodada 34 | Todos | 2022–25 | vs. logística |
|---|---|---|---|---|---|---|---|
| Só gols (original) | 0,127 | 0,086 | 0,057 | 0,048 | 0,0797 | 0,0934 | +0,0080 ± 0,0023 |
| + mercado | 0,107 | 0,081 | 0,057 | 0,050 | 0,0736 | 0,0822 | +0,0019 ± 0,0009 |
| **+ mercado + 4 blocos** | 0,106 | **0,080** | 0,055 | 0,048 | 0,0723 | 0,0817 | +0,0006 ± 0,0005 |
| Logística, Elo do mercado (etapa 1) | **0,104** | 0,080 | **0,055** | **0,048** | **0,0717** | **0,0810** | |
| Histórico por posição | 0,146 | 0,092 | 0,066 | 0,048 | 0,0879 | 0,1016 | |

O que aprendemos:
- **As odds fecharam ~90% da distância** para a logística, quase toda no começo do campeonato (rodada 10: de 0,127 para 0,106). Era exatamente a fraqueza do bayesiano: com poucos gols, a força mal saía do lugar.
- **Os blocos só funcionam junto com o mercado.** Só com gols, a mudança dentro da temporada estimada é ~0,02 (quase nada: os placares são ruidosos demais para enxergá-la). Com as odds, vira ~0,065: o mercado percebe quando um time melhora ou piora no meio do ano.
- **Empate técnico com a logística:** diferença de 1,2 erro-padrão, melhor em 6 de 11 temporadas. Pela nossa regra, ele não substitui a etapa 1. A média dos dois também empata (0,0718), porque agora os dois bebem da mesma fonte (as odds).
- **Bug de convergência corrigido:** com κ ~ Normal(1, 1), 2 dos 44 ajustes tinham uma cadeia presa num "espelho" (κ ≈ −5 e forças quase zeradas; r_hat 1,55). Agora κ ~ LogNormal(0; 0,5), que só aceita valores positivos. Esses dois cenários mudaram até 15 pontos percentuais; os outros, só ruído de Monte Carlo. Os ajustes restantes ficam em r_hat 1,01–1,02 com 500 sorteios, o normal para a versão rápida. Na de 2026 (1.000 sorteios), r_hat fica em 1,006.
- **Calibração:** o modelo ainda exagera um pouco na faixa de 25–50% (previu 38%, aconteceu 27%) e subestima a de 2–10% (5% contra 10%). O padrão é parecido com o da logística.

**Decisão:** a logística com o Elo do mercado continua como **modelo principal**: é mais simples, mais rápida e não foi superada. O bayesiano 2.0 vira a **segunda opinião**: chega ao mesmo nível por outro caminho (gols + odds, sem Elo) e ainda entrega a incerteza de cada força. Quando os dois concordam, a previsão fica mais confiável; quando discordam, vale olhar o porquê.

**2026 após a rodada 33** (bayesiano 2.0 / logística): Grêmio 58,0% / 57,2%, Internacional 46,7% / 44,7%, Vitória 29,6% / 30,1%, Mirassol 27,1% / 28,8%, Corinthians 22,6% / 21,6%, Botafogo 8,0% / 9,8%, Coritiba 4,8% / 4,7%, Vasco 3,4% / 3,3%. No bayesiano original, Corinthians tinha 41% e Vasco 12%: só com gols, o modelo não enxergava o que o mercado vê nesses dois elencos.

## Atualizações por rodada

| Depois da rodada | Grêmio | Internacional | Vitória | Mirassol | Corinthians | Botafogo | Coritiba | Vasco |
|---|---|---|---|---|---|---|---|---|
| 32 | 74,4% | 29,4% | 16,8% | 18,1% | 33,8% | 4,4% | 21,6% | 1,5% |
| 33 | 57,2% | 44,7% | 30,1% | 28,8% | 21,6% | 9,8% | 4,7% | 3,3% |

A partir da rodada 33, o histórico completo (dois modelos, todos os times) fica em `data/processed/historico_previsoes.csv` e no gráfico "Como as chances mudaram" da página. Chapecoense e Remo acima de 99% nas duas. Na rodada 33: Grêmio 2 x 0 Athletico-PR, Corinthians 2 x 0 Mirassol, Coritiba 2 x 0 Vitória, Fluminense 2 x 1 Internacional, Remo 2 x 1 Botafogo.

## Ideias para depois

- xG (gols esperados) e valor de mercado do elenco, se houver fontes com histórico.
- Bayesiano: correção de Dixon-Coles para empates, e um passo de força por rodada em vez de por bloco (mais lento).
- Fator casa específico por time (altitude, calor, viagens longas).

## Diário de práticas

- **Prática 1 – fator casa** (`scripts/pratica_01_fator_casa.ipynb`): a queda de 2020 (sem público) se encaixa numa tendência de queda gradual; 2017, com torcida, teve a menor taxa.
- **Prática 2 – rating Elo** (`scripts/pratica_02_elo.ipynb`): Elo bem calibrado com K = 20 (erro 6,3% menor que um chute constante). A melhor vantagem de casa caiu de ~100 pontos (2013–16) para ~80 (2021–25), confirmando a prática 1. Com K = 40 as notas reagem demais e as probabilidades ficam extremas (excesso de confiança).
