# Quem cai no Brasileirão 2026?

Probabilidade de rebaixamento de cada clube da Série A de 2026, atualizada a cada rodada por **dois modelos independentes**: uma regressão logística sobre um Elo corrigido pelas odds e um modelo hierárquico bayesiano de gols, em PyMC.

**[Ver a página com as previsões atuais](https://tokoyamizinho.github.io/brasileirao-rebaixamento/)**

## Como funciona

1. **Dados:** 5.600+ jogos da Série A desde 2012, com resultados e odds de fechamento ([football-data.co.uk](https://www.football-data.co.uk/)), conferidos placar a placar contra uma segunda fonte.
2. **Força dos times:** um **Elo do mercado**, nota que depois de cada jogo é corrigida em direção ao que as odds esperavam, e não ao resultado. Assim ela absorve o que as casas de apostas sabem (lesões, crises, elenco) e continua valendo para os jogos futuros, que ainda não têm odds.
3. **Cada jogo:** uma regressão logística multinomial transforma a diferença de Elo em probabilidades de vitória, empate e derrota.
4. **O campeonato:** 20.000 simulações Monte Carlo das rodadas restantes, na ordem do calendário, com o Elo se atualizando dentro de cada simulação e desempates oficiais.

O **modelo bayesiano** chega ao mesmo tipo de resposta por outro caminho. Ele estima o ataque e a defesa de cada time (Poisson hierárquica, Baio & Blangiardo 2010), com forças que mudam entre temporadas (AR(1)) e dentro delas, e que precisam explicar ao mesmo tempo os gols e as odds dos jogos passados. Ele serve de segunda opinião e entrega a incerteza de cada força.

## Resultados

**Backtest:** temporadas 2015–2025 "congeladas" em vários pontos (da 10ª à 34ª rodada), simulando o resto só com o que se sabia naquele dia. Brier das probabilidades de rebaixamento (menor é melhor):

| Modelo | Brier | Comparação |
|---|---|---|
| **Logística + Elo do mercado** (principal) | **0,075** | 19% menos erro que o "histórico por posição" |
| Bayesiano, gols + odds (cortes 10, 20, 30, 34) | 0,072 | empate com o principal nos mesmos cortes (0,072) |
| Logística + Elo comum (versão inicial) | 0,082 | precisava de uma incerteza ajustada à mão |
| Histórico por posição ("quantas vezes quem estava aqui caiu") | 0,092 | linha de base |

**Jogo a jogo** (log loss, 2017–2026, sempre fora da amostra): Elo do mercado 1,014, Elo comum 1,024 e as odds do próprio jogo, como referência, 1,000.

**O que foi testado e não entrou:** forma recente, gols sofridos, rodagem dos promovidos, motivação na reta final, confronto direto e clássicos, valor de mercado dos elencos, XGBoost (o Optuna preferiu árvores de uma só pergunta) e pesos por recência. Cada ideia virou uma hipótese com o teste definido antes de ver o resultado. Uma mudança só entrava se errasse menos por mais de 2 erros-padrão e em pelo menos metade das temporadas. O que fez diferença não foi um algoritmo mais sofisticado, e sim uma informação nova: as odds.

Os detalhes de cada teste estão no [diário do projeto](DIARIO.md).

## Como rodar

```bash
python -m pip install -r requirements.txt
python atualizar.py              # baixa os dados, confere, roda os dois modelos e gera a página (~5 min)
```

O `atualizar.py` para e explica o motivo se algo nos dados estiver estranho (time desconhecido, jogo repetido, total diferente de 380). Se não há jogo novo, ele encerra em segundos. Outras opções:

```bash
python atualizar.py --publicar                # no fim, commit + push (atualiza o GitHub Pages)
python atualizar.py --agendar 09:00 --publicar # tarefa diária no Agendador do Windows
python 07_backtest.py                         # backtest do modelo principal
python 05_modelo_bayesiano.py --backtest --amostras 500   # backtest do bayesiano (~1 h)
```

## Estrutura

```
atualizar.py              atualização completa de cada rodada (chama os scripts abaixo)
01_coleta_dados.py        coleta, padronização dos nomes e checagens de qualidade
02_features.py            Elo, Elo do mercado e demais features (Optuna para os hiperparâmetros)
03_modelo_logistico.py    etapa 1: regressão logística com validação walk-forward
04_modelo_xgboost.py      etapa 2: XGBoost (testado, não adotado)
05_modelo_bayesiano.py    etapa 3: modelo hierárquico bayesiano (PyMC + nutpie)
06_simulacao.py           Monte Carlo das rodadas restantes
07_backtest.py            backtest em temporadas passadas
pagina/                   modelo da página; o atualizar.py coloca os números nele
docs/index.html           página gerada (GitHub Pages)
data/manual/              calendário, resultados recentes digitados e valor dos elencos
data/processed/           parâmetros dos modelos, histórico das previsões e resultados
scripts/                  notebooks de prática (fator casa, Elo)
DIARIO.md                 registro completo das decisões e dos testes
```

## Fontes

- [football-data.co.uk](https://www.football-data.co.uk/): resultados e odds da Série A desde 2012.
- [Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset): resultados de 2003 a 2024, para conferência e para a rodagem dos clubes.
- [ValorFinal](https://valorfinal.com.br/brasileirao/2026): calendário de 2026 e resultados antes de entrarem no football-data.
- Valor dos elencos: Transfermarkt, via reportagens da imprensa (fontes por temporada em `data/manual/valor_elenco.csv`).

## Autor

José Eduardo · [github.com/Tokoyamizinho](https://github.com/Tokoyamizinho)

## Licença

Código sob a [licença MIT](LICENSE). Os dados pertencem às fontes acima.
