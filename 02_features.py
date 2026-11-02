"""
02_features.py
==============

Etapa 2 do projeto "Previsão dos rebaixados do Brasileirão 2026".

O QUE ESTE SCRIPT FAZ
---------------------
Para cada partida desde 2012, calcula as features de PRIORIDADE 1 do nosso
plano, sempre usando só o que se sabia ANTES de a bola rolar:

1. Elo de cada time antes do jogo, em duas versões:
   - "simples": como na prática 2 (vitória/empate/derrota);
   - "com placar": goleadas movem mais a nota do que vitórias magras.
2. Rodagem: em quantas das 10 temporadas anteriores o clube jogou a Série A.
   Também marca se o time é recém-promovido.
3. Forma recente: média de pontos, gols marcados e gols sofridos nos
   últimos 5 jogos do time na temporada.

Também salva o "estado atual" de cada time em 2026 (Elo de hoje, forma, rodagem),
que será o ponto de partida da simulação.

COMO RODAR
----------
    python 02_features.py                 # usa os hiperparâmetros salvos (ou os padrões)
    python 02_features.py --otimizar      # escolhe os hiperparâmetros do Elo com Optuna
    python 02_features.py --otimizar --tentativas 300

O --otimizar precisa do Optuna:  python -m pip install optuna

Requisitos: rodar antes o 01_coleta_dados.py --conferir (que baixa também a base
do GitHub, usada aqui para contar a rodagem desde 2003).

ARQUIVOS
--------
Lê:     data/processed/partidas.csv
        data/raw/brasileirao_github_2003_2024.csv
Gera:   data/processed/features_partidas.csv   uma linha por partida, com as features
        data/processed/estado_atual_2026.csv   situação de cada time hoje
        data/processed/params_elo.json         hiperparâmetros do Elo (com --otimizar)
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# Reaproveitamos a padronização de nomes do script 01. Como o nome do arquivo
# começa com número, não dá para usar "import 01_coleta_dados"; o import_module
# resolve isso.
coleta = import_module("01_coleta_dados")

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

PASTA_PROCESSADOS = coleta.PASTA_PROCESSADOS
CAMINHO_GITHUB = coleta.PASTA_RAW / "brasileirao_github_2003_2024.csv"
CAMINHO_PARAMS = PASTA_PROCESSADOS / "params_elo.json"
TEMPORADA_ATUAL = coleta.TEMPORADA_ATUAL

# Hiperparâmetros padrão do Elo (os da prática 2). Se existir params_elo.json,
# gerado pelo --otimizar, os valores de lá substituem estes.
PARAMS_PADRAO = {
    "usar_placar": False,
    "K": 20.0,
    "vantagem_casa": 80.0,
    "regressao": 0.3,
    "rating_promovido": 1450.0,
}

# Elo do mercado (ver calcular_elo, alvo="mercado"). Padrão = valores que o
# Optuna escolheu com 2014–2021; o --otimizar recalcula e salva em
# params_elo_mercado.json.
CAMINHO_PARAMS_MERCADO = PASTA_PROCESSADOS / "params_elo_mercado.json"
PARAMS_MERCADO_PADRAO = {
    "K": 145.0,
    "vantagem_casa": 92.0,
    "regressao": 0.47,
    "rating_promovido": 1354.0,
}

# Separação temporal para escolher os hiperparâmetros (ver --otimizar):
#   AJUSTE: o Optuna só enxerga estes anos para escolher os valores
#   TESTE:  anos "novos", usados apenas para conferir se a escolha generaliza
# 2012–2013 ficam de fora: as notas ainda estão se ajustando ("aquecimento").
TEMPORADAS_AJUSTE = (2014, 2021)
TEMPORADAS_TESTE = (2022, 2025)

# Forma recente
JANELA_FORMA = 5
# Valores "neutros" para o 1º jogo da temporada, quando ainda não há jogos na
# janela. São as médias históricas por time e por jogo na Série A:
#   ~1,2 gol marcado (e sofrido) e ~1,37 ponto [= (3 − proporção de empates) / 2].
VALORES_NEUTROS = {"pontos": 1.37, "gols_marcados": 1.2, "gols_sofridos": 1.2}

PONTOS_REAIS = {"M": 1.0, "E": 0.5, "V": 0.0}


# =============================================================================
# 2. ELO
# =============================================================================

def resultado_esperado(rating_mandante: float, rating_visitante: float, vantagem_casa: float) -> float:
    """Igual à prática 2: número entre 0 e 1 (1 = vitória certa do mandante)."""
    diferenca = (rating_mandante + vantagem_casa) - rating_visitante
    return 1 / (1 + 10 ** (-diferenca / 400))


def multiplicador_placar(saldo: int) -> float:
    """Quanto o tamanho da vitória aumenta o ajuste do Elo.

    É a regra do World Football Elo (ranking de seleções em eloratings.net):
        vitória por 0 ou 1 gol  -> 1,0  (empate também cai aqui)
        vitória por 2 gols      -> 1,5
        vitória por 3+ gols     -> (11 + saldo) / 8    ex.: 3 gols = 1,75; 5 gols = 2,0
    """
    saldo = abs(saldo)
    if saldo <= 1:
        return 1.0
    if saldo == 2:
        return 1.5
    return (11 + saldo) / 8


def calcular_elo(partidas: pd.DataFrame, usar_placar: bool = False, K: float = 20,
                 vantagem_casa: float = 80, regressao: float = 0.3,
                 rating_promovido: float = 1450, alvo: str = "resultado") -> tuple[pd.DataFrame, dict]:
    """Percorre os jogos em ordem e devolve (tabela com Elo pré-jogo, notas atuais).

    É a mesma lógica da prática 2, com três diferenças:
    - opção usar_placar, que multiplica o ajuste pelo multiplicador_placar;
    - opção alvo="mercado" (o "Elo do mercado"): depois de cada jogo, a nota é
      corrigida em direção ao que o MERCADO esperava antes do jogo
      (P(vitória) + 0,5 x P(empate), tirado das odds), e não em direção ao
      resultado. Assim a nota absorve o que as casas de apostas sabem
      (lesões, escalações, crises) e vale também para jogos futuros, que
      ainda não têm odds. Não há vazamento: as odds são conhecidas ANTES do jogo.
      Jogo sem odds = nota não muda.
    - a tabela devolvida mantém o mesmo índice de `partidas`, para podermos
      juntar as colunas com segurança depois.

    `partidas` precisa estar em ordem cronológica.
    """
    if alvo == "mercado":
        alvos = (partidas["prob_mercado_mandante"] + 0.5 * partidas["prob_mercado_empate"]).to_numpy()
    else:
        alvos = partidas["resultado"].map(PONTOS_REAIS).to_numpy()
    alvos = pd.Series(alvos, index=partidas.index)
    ratings: dict[str, float] = {}
    times_temporada_anterior: set[str] = set()
    primeira_temporada = partidas["temporada"].min()
    pre_m, pre_v, esperados = [], [], []

    for temporada, jogos in partidas.groupby("temporada", sort=True):
        # Início da temporada: regressão à média e promovidos
        times = set(jogos["mandante"]) | set(jogos["visitante"])
        for time in times:
            if temporada == primeira_temporada:
                ratings[time] = 1500.0
            elif time in times_temporada_anterior:
                ratings[time] = 1500 + (ratings[time] - 1500) * (1 - regressao)
            else:
                ratings[time] = rating_promovido
        times_temporada_anterior = times

        # Jogo a jogo: primeiro GUARDA a nota pré-jogo, depois atualiza
        for m, v, real, gm, gv in zip(jogos["mandante"], jogos["visitante"], alvos.loc[jogos.index],
                                      jogos["gols_mandante"], jogos["gols_visitante"]):
            esperado = resultado_esperado(ratings[m], ratings[v], vantagem_casa)
            pre_m.append(ratings[m])
            pre_v.append(ratings[v])
            esperados.append(esperado)

            if pd.isna(real):          # sem odds (alvo mercado): não aprende nada
                continue
            ajuste = K * (real - esperado)
            if usar_placar and alvo == "resultado":
                ajuste *= multiplicador_placar(gm - gv)
            ratings[m] += ajuste
            ratings[v] -= ajuste

    saida = pd.DataFrame(index=partidas.index)
    saida["elo_pre_mandante"] = pre_m
    saida["elo_pre_visitante"] = pre_v
    saida["elo_esperado"] = esperados
    return saida, ratings


def erro_quadratico_medio(partidas: pd.DataFrame, elo: pd.DataFrame, periodo: tuple[int, int]) -> float:
    """EQM entre o resultado esperado pelo Elo e o real, só no período pedido."""
    filtro = partidas["temporada"].between(*periodo)
    real = partidas.loc[filtro, "resultado"].map(PONTOS_REAIS)
    return float(((real - elo.loc[filtro, "elo_esperado"]) ** 2).mean())


def otimizar_elo(partidas: pd.DataFrame, tentativas: int) -> dict:
    """Escolhe os hiperparâmetros do Elo com Optuna, para as duas versões
    (simples e com placar), e fica com a que errar menos no período de AJUSTE.

    O período de TESTE é só impresso, para conferirmos se a escolha generaliza.
    Lembra da prática 2? Ganhar no ajuste e perder no teste é sinal de overfitting.
    """
    try:
        import optuna
    except ImportError:
        sys.exit("O --otimizar precisa do Optuna. Instale com: python -m pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n=== Otimizando o Elo ({tentativas} tentativas por versão) ===")
    print(f"  ajuste: {TEMPORADAS_AJUSTE[0]}–{TEMPORADAS_AJUSTE[1]}   "
          f"teste: {TEMPORADAS_TESTE[0]}–{TEMPORADAS_TESTE[1]}")

    melhores = {}
    for usar_placar in (False, True):
        def objetivo(trial):
            params = {
                "K": trial.suggest_float("K", 5, 50),
                "vantagem_casa": trial.suggest_float("vantagem_casa", 0, 150),
                "regressao": trial.suggest_float("regressao", 0, 0.8),
                "rating_promovido": trial.suggest_float("rating_promovido", 1350, 1550),
            }
            elo, _ = calcular_elo(partidas, usar_placar=usar_placar, **params)
            return erro_quadratico_medio(partidas, elo, TEMPORADAS_AJUSTE)

        # seed fixa: rodando de novo, o resultado é o mesmo (reprodutibilidade)
        estudo = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        estudo.optimize(objetivo, n_trials=tentativas)

        params = {"usar_placar": usar_placar, **estudo.best_params}
        elo, _ = calcular_elo(partidas, **params)
        eqm_teste = erro_quadratico_medio(partidas, elo, TEMPORADAS_TESTE)
        nome = "com placar" if usar_placar else "simples"
        print(f"\n  Elo {nome}:")
        print("    " + ", ".join(f"{k}={v:.2f}" for k, v in estudo.best_params.items()))
        print(f"    erro no ajuste: {estudo.best_value:.5f}   erro no teste: {eqm_teste:.5f}")
        melhores[nome] = (estudo.best_value, eqm_teste, params)

    # Referência: os valores padrão, sem otimização
    elo_padrao, _ = calcular_elo(partidas, **PARAMS_PADRAO)
    print(f"\n  Padrão (prática 2): erro no ajuste "
          f"{erro_quadratico_medio(partidas, elo_padrao, TEMPORADAS_AJUSTE):.5f}   "
          f"erro no teste {erro_quadratico_medio(partidas, elo_padrao, TEMPORADAS_TESTE):.5f}")

    # Escolhemos pelo período de AJUSTE (e não pelo teste!), para o teste
    # continuar sendo uma avaliação honesta.
    nome_escolhido = min(melhores, key=lambda n: melhores[n][0])
    print(f"\n  Versão escolhida (menor erro no ajuste): Elo {nome_escolhido}")
    return melhores[nome_escolhido][2]


def otimizar_elo_mercado(partidas: pd.DataFrame, tentativas: int) -> dict:
    """Mesma ideia do otimizar_elo, para o Elo do mercado. O critério continua
    sendo prever os RESULTADOS reais (no período de ajuste): o mercado é só o
    "professor" que corrige a nota jogo a jogo."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        params = {
            # K alto faz sentido aqui: a expectativa do mercado tem bem menos
            # ruído que o resultado do jogo, então dá para confiar mais nela.
            "K": trial.suggest_float("K", 5, 400),
            "vantagem_casa": trial.suggest_float("vantagem_casa", 0, 150),
            "regressao": trial.suggest_float("regressao", 0, 0.8),
            "rating_promovido": trial.suggest_float("rating_promovido", 1150, 1550),
        }
        elo, _ = calcular_elo(partidas, alvo="mercado", **params)
        return erro_quadratico_medio(partidas, elo, TEMPORADAS_AJUSTE)

    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=tentativas)
    elo, _ = calcular_elo(partidas, alvo="mercado", **estudo.best_params)
    print("\n  Elo do mercado:")
    print("    " + ", ".join(f"{k}={v:.2f}" for k, v in estudo.best_params.items()))
    print(f"    erro no ajuste: {estudo.best_value:.5f}   "
          f"erro no teste: {erro_quadratico_medio(partidas, elo, TEMPORADAS_TESTE):.5f}")
    return dict(estudo.best_params)


def carregar_params() -> dict:
    if CAMINHO_PARAMS.exists():
        params = json.loads(CAMINHO_PARAMS.read_text(encoding="utf-8"))
        print(f"  hiperparâmetros lidos de {CAMINHO_PARAMS.name}")
        return params
    print("  usando hiperparâmetros padrão (rode com --otimizar para escolhê-los com Optuna)")
    return dict(PARAMS_PADRAO)


def carregar_params_mercado() -> dict:
    if CAMINHO_PARAMS_MERCADO.exists():
        return json.loads(CAMINHO_PARAMS_MERCADO.read_text(encoding="utf-8"))
    return dict(PARAMS_MERCADO_PADRAO)


# =============================================================================
# 3. RODAGEM NA SÉRIE A
# =============================================================================

def temporadas_na_serie_a(partidas: pd.DataFrame) -> pd.DataFrame:
    """Tabela (temporada, time) com todos os participantes da Série A, de 2003
    até hoje: 2003–2011 vêm da base do GitHub e 2012 em diante do football-data."""
    github = pd.read_csv(CAMINHO_GITHUB)
    datas = pd.to_datetime(github["data"], dayfirst=True, errors="coerce")
    nao_reconhecidos: set = set()
    github = pd.DataFrame({
        # jogos de jan–mar contam para a temporada anterior (ex.: 2020 terminou em 2021)
        "temporada": datas.dt.year - (datas.dt.month <= 3).astype(int),
        "time": github["mandante"].apply(coleta.padronizar_nome_time, nao_reconhecidos=nao_reconhecidos),
    })
    if nao_reconhecidos:
        print(f"  AVISO: nomes não reconhecidos na base do GitHub: {sorted(nao_reconhecidos)}")
    github = github[github["temporada"] < partidas["temporada"].min()]

    recentes = partidas[["temporada", "mandante"]].rename(columns={"mandante": "time"})
    return pd.concat([github, recentes]).drop_duplicates().reset_index(drop=True)


def calcular_rodagem(participantes: pd.DataFrame, anos: int = 10) -> pd.DataFrame:
    """Para cada (temporada, time): em quantas das `anos` temporadas anteriores
    o time esteve na Série A, e se é recém-promovido (não estava no ano anterior).

    Observação: a base começa em 2003, então para temporadas até 2012 a janela
    de 10 anos fica incompleta (em 2012, por exemplo, só cabem 2003–2011).
    """
    presentes = set(zip(participantes["temporada"], participantes["time"]))
    linhas = []
    for temporada, time in presentes:
        anteriores = range(temporada - anos, temporada)
        linhas.append({
            "temporada": temporada,
            "time": time,
            "rodagem": sum((t, time) in presentes for t in anteriores),
            "promovido": int((temporada - 1, time) not in presentes),
        })
    return pd.DataFrame(linhas)


# =============================================================================
# 4. FORMA RECENTE
# =============================================================================

def formato_longo(partidas: pd.DataFrame) -> pd.DataFrame:
    """Cada partida vira duas linhas, uma por time, do ponto de vista dele.
    (É o mesmo truque do script 01, para numerar os jogos.)"""
    comum = ["temporada", "data"]
    mand = partidas[comum].copy()
    mand["id_partida"] = partidas.index
    mand["lado"] = "mandante"
    mand["time"] = partidas["mandante"]
    mand["gols_marcados"] = partidas["gols_mandante"]
    mand["gols_sofridos"] = partidas["gols_visitante"]

    visi = partidas[comum].copy()
    visi["id_partida"] = partidas.index
    visi["lado"] = "visitante"
    visi["time"] = partidas["visitante"]
    visi["gols_marcados"] = partidas["gols_visitante"]
    visi["gols_sofridos"] = partidas["gols_mandante"]

    longo = pd.concat([mand, visi], ignore_index=True)
    longo["pontos"] = 0
    longo.loc[longo["gols_marcados"] > longo["gols_sofridos"], "pontos"] = 3
    longo.loc[longo["gols_marcados"] == longo["gols_sofridos"], "pontos"] = 1
    return longo.sort_values(["temporada", "time", "data", "id_partida"]).reset_index(drop=True)


def calcular_forma(partidas: pd.DataFrame, janela: int = JANELA_FORMA) -> pd.DataFrame:
    """Médias dos últimos `janela` jogos ANTERIORES de cada time, na temporada.

    O detalhe que evita vazamento é o shift(1): ele "empurra" os valores uma
    linha para baixo, de modo que a média de um jogo só enxerga os jogos
    anteriores a ele, nunca o próprio jogo.
    """
    longo = formato_longo(partidas)
    grupo = longo.groupby(["temporada", "time"])
    for coluna in ["pontos", "gols_marcados", "gols_sofridos"]:
        anteriores = grupo[coluna].shift(1)
        longo[f"{coluna}_{janela}"] = (
            anteriores.groupby([longo["temporada"], longo["time"]])
            .rolling(janela, min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
        )
        # 1º jogo da temporada: ainda não há janela -> valor neutro
        longo[f"{coluna}_{janela}"] = longo[f"{coluna}_{janela}"].fillna(VALORES_NEUTROS[coluna])
    # Quantos jogos de fato entraram na média (0 no 1º jogo, até `janela`)
    longo["jogos_janela"] = grupo.cumcount().clip(upper=janela)
    return longo


# =============================================================================
# 4b. MOTIVAÇÃO (prioridade 2)
# =============================================================================
# Hipótese: na reta final, times ameaçados de queda se esforçam mais, e times
# sem chance de escapar ("desenganados") relaxam. A análise exploratória
# (últimos 10 jogos, 2017–2025) mostrou: ameaçados fazem ~0,08 ponto/jogo a mais
# do que o modelo espera; desenganados ~0,20 a menos; no 1º turno, nada.

RETA_FINAL = 10            # só vale nos últimos 10 jogos do time
FATOR_DESENGANADO = 1.5    # desenganado: está mais de 1,5 ponto/jogo restante abaixo da salvação


def folga_z4(pontos, p16, p17):
    """Distância para a zona de rebaixamento.
    Fora do Z4: pontos acima do 17º (positivo).
    No Z4: pontos que faltam para alcançar o 16º (zero ou negativo).
    Funciona com números ou com matrizes do numpy (usado também na simulação)."""
    return np.where(pontos > p17, pontos - p17, pontos - p16)


def indicadores_motivacao(folga, jogos_restantes):
    """Devolve (na_briga, desenganado), cada um 0 ou 1."""
    reta = jogos_restantes <= RETA_FINAL
    desenganado = reta & (folga <= -FATOR_DESENGANADO * jogos_restantes)
    na_briga = reta & ~desenganado & (folga <= jogos_restantes)
    return na_briga.astype(int), desenganado.astype(int)


def calcular_motivacao(partidas: pd.DataFrame) -> pd.DataFrame:
    """Situação de cada time na tabela ANTES de cada jogo.

    Usamos só jogos de DATAS anteriores: jogos do mesmo dia acontecem ao mesmo
    tempo, então um não pode "saber" o resultado do outro.
    """
    saida = pd.DataFrame(index=partidas.index,
                         columns=["folga_z4_mandante", "folga_z4_visitante",
                                  "jogos_restantes_mandante", "jogos_restantes_visitante"], dtype=float)
    for _, jogos in partidas.groupby("temporada"):
        times = sorted(set(jogos["mandante"]) | set(jogos["visitante"]))
        total_jogos = 2 * (len(times) - 1)            # 38 com 20 times
        pontos = {t: 0 for t in times}
        disputados = {t: 0 for t in times}
        for _, do_dia in jogos.groupby("data", sort=True):
            ordenados = sorted(pontos.values(), reverse=True)
            p16, p17 = ordenados[15], ordenados[16]
            for i, j in do_dia.iterrows():
                for lado in ("mandante", "visitante"):
                    time_ = j[lado]
                    saida.loc[i, f"folga_z4_{lado}"] = folga_z4(pontos[time_], p16, p17)
                    saida.loc[i, f"jogos_restantes_{lado}"] = total_jogos - disputados[time_]
            for _, j in do_dia.iterrows():               # só depois atualiza a tabela
                gm, gv = j["gols_mandante"], j["gols_visitante"]
                pontos[j["mandante"]] += 3 if gm > gv else (1 if gm == gv else 0)
                pontos[j["visitante"]] += 3 if gv > gm else (1 if gm == gv else 0)
                disputados[j["mandante"]] += 1
                disputados[j["visitante"]] += 1
    for lado in ("mandante", "visitante"):
        briga, deseng = indicadores_motivacao(saida[f"folga_z4_{lado}"].to_numpy(),
                                              saida[f"jogos_restantes_{lado}"].to_numpy())
        saida[f"na_briga_{lado}"] = briga
        saida[f"desenganado_{lado}"] = deseng
    return saida


# =============================================================================
# 5. MONTAGEM FINAL
# =============================================================================

def montar_features(partidas: pd.DataFrame, params: dict,
                    params_mercado: dict) -> tuple[pd.DataFrame, dict, dict]:
    # Elo e forma são juntados pelo ÍNDICE (o número da linha em `partidas`),
    # que é o mesmo nas duas tabelas. Por isso fazemos esses joins primeiro:
    # o merge da rodagem, mais abaixo, cria um índice novo.
    elo, ratings = calcular_elo(partidas, **params)
    tabela = partidas.join(elo)
    tabela["elo_diff"] = tabela["elo_pre_mandante"] - tabela["elo_pre_visitante"]

    elo_m, ratings_mercado = calcular_elo(partidas, alvo="mercado", **params_mercado)
    tabela["elo_mercado_pre_mandante"] = elo_m["elo_pre_mandante"]
    tabela["elo_mercado_pre_visitante"] = elo_m["elo_pre_visitante"]
    tabela["elo_mercado_diff"] = tabela["elo_mercado_pre_mandante"] - tabela["elo_mercado_pre_visitante"]

    # Forma: o formato longo tem uma linha por (partida, lado); voltamos para
    # uma linha por partida com colunas _mandante e _visitante.
    forma = calcular_forma(partidas)
    colunas_forma = [f"pontos_{JANELA_FORMA}", f"gols_marcados_{JANELA_FORMA}",
                     f"gols_sofridos_{JANELA_FORMA}", "jogos_janela"]
    largo = forma.pivot(index="id_partida", columns="lado", values=colunas_forma)
    largo.columns = [f"{col}_{lado}" for col, lado in largo.columns]
    tabela = tabela.join(largo)
    tabela["jogos_janela_mandante"] = tabela["jogos_janela_mandante"].astype(int)
    tabela["jogos_janela_visitante"] = tabela["jogos_janela_visitante"].astype(int)

    # Motivação: situação na tabela antes do jogo (mesmo índice, então join direto)
    tabela = tabela.join(calcular_motivacao(partidas))

    # Rodagem: junta pelo (temporada, time) do mandante e depois do visitante
    rodagem = calcular_rodagem(temporadas_na_serie_a(partidas))
    for lado in ["mandante", "visitante"]:
        tabela = tabela.merge(
            rodagem.rename(columns={"time": lado, "rodagem": f"rodagem_{lado}",
                                    "promovido": f"promovido_{lado}"}),
            on=["temporada", lado], how="left")

    return tabela, ratings, ratings_mercado


def estado_atual(tabela: pd.DataFrame, ratings: dict, partidas: pd.DataFrame,
                 ratings_mercado: dict) -> pd.DataFrame:
    """Situação de cada time de 2026 ANTES do próximo jogo: Elo de hoje,
    forma dos últimos jogos (já incluindo o mais recente) e rodagem."""
    atual = partidas[partidas["temporada"] == TEMPORADA_ATUAL]
    longo = formato_longo(atual)
    ultimos = longo.groupby("time").tail(JANELA_FORMA).groupby("time").agg(
        jogos_janela=("pontos", "size"),
        **{f"pontos_{JANELA_FORMA}": ("pontos", "mean"),
           f"gols_marcados_{JANELA_FORMA}": ("gols_marcados", "mean"),
           f"gols_sofridos_{JANELA_FORMA}": ("gols_sofridos", "mean")},
    )
    jogos = longo.groupby("time").size().rename("jogos")
    estado = (pd.DataFrame({"elo": pd.Series(ratings), "elo_mercado": pd.Series(ratings_mercado)})
              .join(jogos, how="inner").join(ultimos))

    rod = tabela[tabela["temporada"] == TEMPORADA_ATUAL]
    rod = pd.concat([
        rod[["mandante", "rodagem_mandante", "promovido_mandante"]].set_axis(["time", "rodagem", "promovido"], axis=1),
        rod[["visitante", "rodagem_visitante", "promovido_visitante"]].set_axis(["time", "rodagem", "promovido"], axis=1),
    ]).drop_duplicates("time").set_index("time")
    estado = estado.join(rod).sort_values("elo")
    estado.index.name = "time"
    return estado.round(3)


# =============================================================================
# 6. CHECAGENS
# =============================================================================

def checar(tabela: pd.DataFrame) -> None:
    """Testes de sanidade. Se algum falhar, o script para com uma mensagem clara."""
    print("\n=== Checagens ===")
    colunas = [c for c in tabela.columns if c.startswith(("elo_", "rodagem_", "promovido_",
                                                           "pontos_", "gols_marcados_", "gols_sofridos_",
                                                           "folga_z4_", "na_briga_", "desenganado_"))]
    faltando = tabela[colunas].isna().sum()
    assert faltando.sum() == 0, f"Há valores faltando:\n{faltando[faltando > 0]}"
    print("  Nenhum valor faltando nas features: OK")

    # Vazamento: no 1º jogo da temporada, a forma tem que ser o valor neutro
    primeiro = tabela[tabela["jogo_n_mandante"] == 1]
    assert (primeiro["jogos_janela_mandante"] == 0).all()
    assert (primeiro[f"pontos_{JANELA_FORMA}_mandante"] == VALORES_NEUTROS["pontos"]).all()
    print("  1º jogo da temporada sem informação de jogos futuros: OK")

    # Rodagem: quem NÃO é promovido estava na Série A no ano anterior,
    # então tem rodagem de pelo menos 1
    assert (tabela.loc[tabela["promovido_mandante"] == 0, "rodagem_mandante"] >= 1).all()
    print("  Rodagem coerente com promovidos: OK")

    # Motivação: só pode valer nos últimos RETA_FINAL jogos, e ninguém é
    # "na briga" e "desenganado" ao mesmo tempo
    fora_da_reta = tabela["jogos_restantes_mandante"] > RETA_FINAL
    assert (tabela.loc[fora_da_reta, ["na_briga_mandante", "desenganado_mandante"]] == 0).all().all()
    assert not ((tabela["na_briga_mandante"] == 1) & (tabela["desenganado_mandante"] == 1)).any()
    print(f"  Motivação só na reta final e sem categorias sobrepostas: OK "
          f"({tabela['na_briga_mandante'].mean():.1%} dos mandantes 'na briga', "
          f"{tabela['desenganado_mandante'].mean():.1%} 'desenganados')")

    # Elo: a soma das notas é conservada dentro da temporada (o que um ganha, o outro perde).
    # Checamos indiretamente: a média do Elo pré-jogo fica perto de 1500.
    media = tabela[["elo_pre_mandante", "elo_pre_visitante"]].stack().mean()
    print(f"  Média geral do Elo pré-jogo: {media:.1f} (esperado: perto de 1500)")


# =============================================================================
# 7. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula as features das partidas.")
    parser.add_argument("--otimizar", action="store_true",
                        help="escolhe os hiperparâmetros do Elo com Optuna e salva em params_elo.json")
    parser.add_argument("--tentativas", type=int, default=200,
                        help="número de tentativas do Optuna por versão do Elo (padrão: 200)")
    args = parser.parse_args()

    print("=== 1. Carregando partidas ===")
    partidas = pd.read_csv(PASTA_PROCESSADOS / "partidas.csv", parse_dates=["data"])
    partidas = partidas.sort_values(["temporada", "data", "id_partida"]).reset_index(drop=True)
    print(f"  {len(partidas)} partidas, {partidas['temporada'].min()}–{partidas['temporada'].max()}")
    if not CAMINHO_GITHUB.exists():
        sys.exit("Falta a base do GitHub. Rode antes: python 01_coleta_dados.py --conferir")

    print("\n=== 2. Hiperparâmetros do Elo ===")
    if args.otimizar:
        params = otimizar_elo(partidas, args.tentativas)
        CAMINHO_PARAMS.write_text(json.dumps(params, indent=2), encoding="utf-8")
        params_mercado = otimizar_elo_mercado(partidas, args.tentativas)
        CAMINHO_PARAMS_MERCADO.write_text(json.dumps(params_mercado, indent=2), encoding="utf-8")
        print(f"  salvos em {CAMINHO_PARAMS.name} e {CAMINHO_PARAMS_MERCADO.name}")
    else:
        params = carregar_params()
        params_mercado = carregar_params_mercado()
    print("  Elo:         " + ", ".join(f"{k}={v}" if isinstance(v, bool) else f"{k}={v:.2f}" for k, v in params.items()))
    print("  Elo mercado: " + ", ".join(f"{k}={v:.2f}" for k, v in params_mercado.items()))

    print("\n=== 3. Calculando features ===")
    tabela, ratings, ratings_mercado = montar_features(partidas, params, params_mercado)
    print(f"  {len(tabela)} partidas x {tabela.shape[1]} colunas")

    checar(tabela)

    estado = estado_atual(tabela, ratings, partidas, ratings_mercado)
    print(f"\n=== 4. Os 8 últimos pelo Elo em {TEMPORADA_ATUAL} ===")
    print(estado.head(8).to_string())

    print("\n=== 5. Salvando ===")
    tabela.to_csv(PASTA_PROCESSADOS / "features_partidas.csv", index=False)
    estado.to_csv(PASTA_PROCESSADOS / f"estado_atual_{TEMPORADA_ATUAL}.csv")
    print(f"  arquivos salvos em {PASTA_PROCESSADOS}")


if __name__ == "__main__":
    main()
