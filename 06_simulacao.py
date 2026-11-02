"""
06_simulacao.py
===============

Simulação Monte Carlo do restante do Brasileirão 2026.

A IDEIA
-------
Não dá para saber como vão terminar os jogos que faltam, mas o modelo nos dá a
CHANCE de cada resultado. Então "jogamos" o resto do campeonato no computador
milhares de vezes, sorteando cada jogo de acordo com essas chances, e contamos:
    em quantas das N simulações o time X terminou entre os 4 últimos?
Essa proporção é a probabilidade de rebaixamento.

COMO CADA SIMULAÇÃO FUNCIONA
----------------------------
Os jogos restantes são percorridos NA ORDEM DO CALENDÁRIO. Para cada jogo:
  1. o modelo logístico calcula P(M), P(E), P(V) a partir da diferença de Elo
     ATUAL daquela simulação;
  2. sorteamos o resultado (M, E ou V);
  3. sorteamos um placar compatível com o resultado (ver abaixo);
  4. atualizamos a tabela (pontos, vitórias, saldo, gols) e o Elo dos dois times.
É a "simulação dinâmica" que combinamos: se numa simulação o Remo vence três
seguidas, o Elo dele sobe e as chances dele nos jogos seguintes melhoram.
Isso representa melhor a incerteza do que congelar o Elo de hoje
(compare rodando com --estatico).

POR QUE SORTEAR O PLACAR?
-------------------------
O modelo só diz M/E/V, mas precisamos de gols por dois motivos: os critérios de
desempate (saldo, gols pró) e o Elo, que considera o tamanho da vitória.
Sorteamos o placar a partir dos placares REAIS das últimas temporadas, dado o
resultado. Ex.: se saiu "vitória do mandante", sorteamos entre 1x0, 2x0, 2x1...
com a frequência com que cada um aconteceu de verdade.

RESULTADOS JÁ CONHECIDOS
------------------------
O football-data demora alguns dias para atualizar. Jogos já disputados que ainda
não estão no arquivo podem ser digitados em data/manual/resultados_recentes_2026.csv.
Eles entram na simulação com o placar fixo (sem sorteio). Quando o football-data
atualizar, eles são ignorados automaticamente, para não contar duas vezes.

COMO RODAR
----------
    python 06_simulacao.py                    # 20.000 simulações
    python 06_simulacao.py --n 50000          # mais simulações, menos erro de sorteio
    python 06_simulacao.py --estatico         # congela o Elo de hoje (comparação)
    python 06_simulacao.py --incerteza-elo 75 # sorteia a força inicial (ver abaixo)

INCERTEZA SOBRE A FORÇA DOS TIMES (--incerteza-elo, padrão 0)
-------------------------------------------------------------
O Elo é uma ESTIMATIVA da força de cada time, não a força verdadeira. Com
--incerteza-elo 75, cada simulação começa com o Elo de cada time somado a um
sorteio de uma normal com desvio padrão de 75 pontos. Isso foi necessário com o
Elo comum; com o Elo do mercado (modelo atual), o backtest mostrou que não é.

FEATURES DE MOTIVAÇÃO
---------------------
Se o modelo salvo usar na_briga_diff / desenganado_diff, a simulação calcula a
situação de cada time na tabela DE CADA SIMULAÇÃO antes de cada data de jogos.

Lê:   data/processed/partidas.csv, estado_atual_2026.csv, params_elo.json,
      modelo_logistico.joblib, jogos_restantes_2026.csv
      data/manual/calendario_2026.csv, resultados_recentes_2026.csv
Gera: data/processed/simulacao_2026.csv          probabilidades por time
      data/processed/simulacao_posicoes_2026.csv distribuição das posições finais
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from importlib import import_module

import joblib
import numpy as np
import pandas as pd

# folga_z4 e indicadores_motivacao: as MESMAS funções usadas para criar as features
features = import_module("02_features")

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"
MANUAL = PASTA / "data" / "manual"
TEMPORADA = 2026
N_REBAIXADOS = 4

# Temporadas usadas para a distribuição de placares (recentes: o número de gols
# por jogo muda um pouco com o tempo)
TEMPORADAS_PLACARES = (2021, 2025)

SEMENTE = 2026   # semente fixa: rodar de novo dá exatamente o mesmo resultado

# Incerteza sobre a força dos times (desvio padrão, em pontos de Elo).
# Histórico da escolha (ver README):
#  - com o Elo comum, 75 corrigia o excesso de confiança do modelo;
#  - com o Elo do mercado (modelo atual), o backtest saiu calibrado já com 0
#    (quem recebe 50–90% caiu em 67% dos casos, previsto 69%), e qualquer
#    incerteza extra deixou o modelo inseguro demais. A nota do mercado já é
#    uma estimativa melhor da força, então não precisa da "folga" manual.
INCERTEZA_PADRAO = 0.0

# K da nota do mercado nos jogos SIMULADOS (que não têm odds; a nota passa a
# ser corrigida pelos resultados sorteados). None = usa o K do Elo comum.
K_SIMULACAO_MERCADO = None


# =============================================================================
# 2. CARREGAR O PONTO DE PARTIDA
# =============================================================================

def tabela_atual(partidas_2026: pd.DataFrame) -> pd.DataFrame:
    """Pontos, jogos, vitórias, gols pró e contra de cada time até agora."""
    linhas = []
    for _, j in partidas_2026.iterrows():
        gm, gv = j["gols_mandante"], j["gols_visitante"]
        linhas.append((j["mandante"], gm, gv))
        linhas.append((j["visitante"], gv, gm))
    t = pd.DataFrame(linhas, columns=["time", "gp", "gc"])
    t["vitorias"] = (t["gp"] > t["gc"]).astype(int)
    t["pontos"] = 3 * t["vitorias"] + (t["gp"] == t["gc"]).astype(int)
    return t.groupby("time").agg(jogos=("gp", "size"), pontos=("pontos", "sum"),
                                 vitorias=("vitorias", "sum"), gols_pro=("gp", "sum"),
                                 gols_contra=("gc", "sum"))


def montar_jogos_a_simular() -> pd.DataFrame:
    """Lista ordenada dos jogos que faltam no nosso arquivo, já marcando os que
    têm resultado conhecido (arquivo manual)."""
    restantes = pd.read_csv(PROCESSADOS / f"jogos_restantes_{TEMPORADA}.csv")
    calendario = pd.read_csv(MANUAL / f"calendario_{TEMPORADA}.csv", parse_dates=["data"])
    jogos = restantes.merge(calendario, on=["mandante", "visitante"], how="left")

    faltando = jogos["rodada"].isna()
    if faltando.any():
        raise SystemExit("Jogos restantes sem lugar no calendário:\n"
                         + jogos.loc[faltando, ["mandante", "visitante"]].to_string(index=False))

    # Jogo adiado sem data (Chapecoense x Vasco): vai para o fim, como decidido
    jogos["data_ordem"] = jogos["data"].fillna(pd.Timestamp(f"{TEMPORADA}-12-31"))

    caminho_manual = MANUAL / f"resultados_recentes_{TEMPORADA}.csv"
    jogos["gols_mandante"] = np.nan
    jogos["gols_visitante"] = np.nan
    if caminho_manual.exists():
        manual = pd.read_csv(caminho_manual)
        conhecidos = jogos.merge(manual[["mandante", "visitante", "gols_mandante", "gols_visitante"]],
                                 on=["mandante", "visitante"], how="left", suffixes=("_x", ""))
        jogos["gols_mandante"] = conhecidos["gols_mandante"].to_numpy()
        jogos["gols_visitante"] = conhecidos["gols_visitante"].to_numpy()
        usados = jogos["gols_mandante"].notna().sum()
        ignorados = len(manual) - usados
        print(f"  resultados manuais usados: {usados}"
              + (f" (ignorados {ignorados}: já estão no football-data)" if ignorados else ""))

    jogos["conhecido"] = jogos["gols_mandante"].notna()
    return jogos.sort_values(["data_ordem", "rodada"]).reset_index(drop=True)


def distribuicao_placares(partidas: pd.DataFrame, temporadas: tuple[int, int] = TEMPORADAS_PLACARES) -> dict:
    """Para cada resultado (M, E, V): lista de placares possíveis e frequências."""
    recentes = partidas[partidas["temporada"].between(*temporadas)]
    dist = {}
    for resultado, grupo in recentes.groupby("resultado"):
        contagem = grupo.groupby(["gols_mandante", "gols_visitante"]).size()
        dist[resultado] = (np.array(contagem.index.tolist()), (contagem / contagem.sum()).to_numpy())
    return dist


# =============================================================================
# 3. A SIMULAÇÃO
# =============================================================================

def multiplicador_placar(saldo: np.ndarray) -> np.ndarray:
    """Mesma regra do 02_features.py, em versão vetorizada (para N simulações de uma vez)."""
    s = np.abs(saldo)
    return np.where(s <= 1, 1.0, np.where(s == 2, 1.5, (11 + s) / 8))


COLUNAS_SUPORTADAS = {"elo_diff", "elo_mercado_diff", "na_briga_diff", "desenganado_diff"}


def simular(n: int, estatico: bool = False, semente: int = SEMENTE, incerteza_elo: float = 0.0) -> dict:
    """Prepara o ponto de partida de 2026 a partir dos arquivos e chama o
    núcleo da simulação (simular_jogos)."""
    rng = np.random.default_rng(semente)

    partidas = pd.read_csv(PROCESSADOS / "partidas.csv", parse_dates=["data"])
    estado = pd.read_csv(PROCESSADOS / f"estado_atual_{TEMPORADA}.csv", index_col="time")
    params = json.loads((PROCESSADOS / "params_elo.json").read_text(encoding="utf-8"))
    pacote = joblib.load(PROCESSADOS / "modelo_logistico.joblib")
    if set(pacote["colunas"]) - COLUNAS_SUPORTADAS:
        raise SystemExit(f"A simulação não sabe calcular {set(pacote['colunas']) - COLUNAS_SUPORTADAS}.")

    tabela = tabela_atual(partidas[partidas["temporada"] == TEMPORADA])
    jogos = montar_jogos_a_simular()
    sim = simular_jogos(tabela, estado["elo"], jogos, pacote["modelo"], params,
                        distribuicao_placares(partidas), n, rng, estatico,
                        colunas=pacote["colunas"], incerteza_elo=incerteza_elo,
                        elo_mercado_inicial=estado.get("elo_mercado"),
                        params_mercado={**features.carregar_params_mercado(),
                                        "K_simulacao": K_SIMULACAO_MERCADO})
    sim.update({"tabela": tabela, "jogos": jogos, "n": n})
    return sim


def posicoes_finais(pontos, vitorias, gols_pro, gols_contra, rng) -> np.ndarray:
    """Classificação final de cada simulação (matriz N x times, valores 1 a 20).

    Critérios de desempate: pontos, vitórias, saldo, gols pró e, persistindo o
    empate, sorteio. Juntamos tudo em uma "nota" só, com cada critério numa
    "casa decimal" diferente."""
    saldo = gols_pro - gols_contra
    nota = (pontos * 1e9 + vitorias * 1e6 + (saldo + 500) * 1e3 + gols_pro
            + rng.random(pontos.shape))   # parte fracionária = sorteio
    ordem = np.argsort(-nota, axis=1)              # índices dos times do 1º ao último
    posicao = np.empty_like(ordem)
    posicao[np.arange(len(pontos))[:, None], ordem] = np.arange(1, pontos.shape[1] + 1)
    return posicao


def simular_jogos(tabela: pd.DataFrame, elo_inicial: pd.Series, jogos: pd.DataFrame,
                  modelo, params: dict, placares: dict, n: int,
                  rng: np.random.Generator, estatico: bool = False,
                  colunas: list[str] = ("elo_diff",), incerteza_elo: float = 0.0,
                  elo_mercado_inicial: pd.Series | None = None,
                  params_mercado: dict | None = None) -> dict:
    """NÚCLEO DA SIMULAÇÃO. Não lê arquivos: recebe tudo pronto. Assim o mesmo
    código simula 2026 (este script) e as temporadas passadas (07_backtest.py).

    tabela:        jogos, pontos, vitorias, gols_pro, gols_contra de cada time no ponto de partida
    elo_inicial:   Elo de cada time no ponto de partida
    jogos:         jogos a simular, EM ORDEM, com colunas mandante, visitante, conhecido,
                   gols_mandante, gols_visitante (os dois últimos só para os conhecidos)
                   e data_ordem (para saber quais jogos acontecem no mesmo dia)
    colunas:       features que o modelo usa
    incerteza_elo: desvio padrão do sorteio somado ao Elo inicial (0 = sem sorteio)
    elo_mercado_inicial, params_mercado: só para modelos com elo_mercado_diff.

    ELO DO MERCADO NA SIMULAÇÃO: jogos futuros não têm odds, então a nota do
    mercado não pode mais ser corrigida pelo mercado. Depois do ponto de
    partida, ela passa a ser corrigida pelos RESULTADOS sorteados, com o K do
    Elo comum (como o próprio mercado faria ao ver os resultados).
    """
    times = sorted(tabela.index)
    idx = {t: i for i, t in enumerate(times)}
    n_times = len(times)
    usa_motivacao = bool({"na_briga_diff", "desenganado_diff"} & set(colunas))
    usa_mercado = "elo_mercado_diff" in colunas
    total_jogos = 2 * (n_times - 1)

    # Todas as N simulações andam juntas: cada variável é uma matriz
    # (N simulações x 20 times). Muito mais rápido que um loop por simulação.
    def matriz(coluna):
        return np.tile(tabela.loc[times, coluna].to_numpy(dtype=float), (n, 1))
    pontos, vitorias = matriz("pontos"), matriz("vitorias")
    gols_pro, gols_contra = matriz("gols_pro"), matriz("gols_contra")
    elo = np.tile(elo_inicial.loc[times].to_numpy(dtype=float), (n, 1))
    if incerteza_elo > 0:
        elo += rng.normal(0, incerteza_elo, size=elo.shape)
    if usa_mercado:
        elo_m = np.tile(elo_mercado_inicial.loc[times].to_numpy(dtype=float), (n, 1))
        if incerteza_elo > 0:
            elo_m += rng.normal(0, incerteza_elo, size=elo_m.shape)
    classes = list(modelo.classes_)            # ordem das colunas do predict_proba
    iM, iE = classes.index("M"), classes.index("E")

    disputados = tabela.loc[times, "jogos"].to_numpy(dtype=int).copy() if usa_motivacao else None
    data_anterior = None

    for _, jogo in jogos.iterrows():
        m, v = idx[jogo["mandante"]], idx[jogo["visitante"]]

        # Motivação: "foto" da tabela no início de cada data (jogos do mesmo
        # dia não enxergam uns aos outros, igual ao 02_features.py)
        if usa_motivacao and jogo["data_ordem"] != data_anterior:
            data_anterior = jogo["data_ordem"]
            ordenados = np.sort(pontos, axis=1)                 # crescente
            p16 = ordenados[:, n_times - 16]                    # 16º colocado
            p17 = ordenados[:, n_times - 17]                    # 17º colocado
            foto_pontos, foto_disputados = pontos.copy(), disputados.copy()

        if jogo["conhecido"]:
            gm = np.full(n, int(jogo["gols_mandante"]))
            gv = np.full(n, int(jogo["gols_visitante"]))
        else:
            # 1) probabilidades pelo modelo, com o Elo atual de cada simulação
            valores = {"elo_diff": elo[:, m] - elo[:, v]}
            if usa_mercado:
                valores["elo_mercado_diff"] = elo_m[:, m] - elo_m[:, v]
            if usa_motivacao:
                ind = {}
                for lado, t in (("m", m), ("v", v)):
                    folga = features.folga_z4(foto_pontos[:, t], p16, p17)
                    restantes = np.full(n, total_jogos - foto_disputados[t])
                    ind[lado] = features.indicadores_motivacao(folga, restantes)
                valores["na_briga_diff"] = ind["m"][0] - ind["v"][0]
                valores["desenganado_diff"] = ind["m"][1] - ind["v"][1]
            x = pd.DataFrame({c: valores[c] for c in colunas})
            p = modelo.predict_proba(x)
            # 2) sorteia o resultado: u < P(M) -> M; u < P(M)+P(E) -> E; senão V
            u = rng.random(n)
            resultado = np.where(u < p[:, iM], "M", np.where(u < p[:, iM] + p[:, iE], "E", "V"))
            # 3) sorteia o placar dado o resultado
            gm = np.empty(n, dtype=int)
            gv = np.empty(n, dtype=int)
            for r in ("M", "E", "V"):
                sel = resultado == r
                if sel.any():
                    opcoes, probs = placares[r]
                    escolha = rng.choice(len(opcoes), size=sel.sum(), p=probs)
                    gm[sel], gv[sel] = opcoes[escolha, 0], opcoes[escolha, 1]

        # 4a) atualiza a tabela
        venceu_m, empate, venceu_v = gm > gv, gm == gv, gm < gv
        pontos[:, m] += 3 * venceu_m + empate
        pontos[:, v] += 3 * venceu_v + empate
        vitorias[:, m] += venceu_m
        vitorias[:, v] += venceu_v
        gols_pro[:, m] += gm
        gols_contra[:, m] += gv
        gols_pro[:, v] += gv
        gols_contra[:, v] += gm
        if usa_motivacao:
            disputados[m] += 1
            disputados[v] += 1

        # 4b) atualiza o Elo (mesma fórmula do 02_features.py)
        if not estatico or jogo["conhecido"]:
            esperado = 1 / (1 + 10 ** (-(elo[:, m] + params["vantagem_casa"] - elo[:, v]) / 400))
            real = venceu_m + 0.5 * empate
            ajuste = params["K"] * (real - esperado)
            if params.get("usar_placar"):
                ajuste *= multiplicador_placar(gm - gv)
            elo[:, m] += ajuste
            elo[:, v] -= ajuste
            if usa_mercado:
                esperado = 1 / (1 + 10 ** (-(elo_m[:, m] + params_mercado["vantagem_casa"] - elo_m[:, v]) / 400))
                k_sim = params_mercado.get("K_simulacao") or params["K"]
                ajuste = k_sim * (real - esperado)
                if params.get("usar_placar"):
                    ajuste *= multiplicador_placar(gm - gv)
                elo_m[:, m] += ajuste
                elo_m[:, v] -= ajuste

    posicao = posicoes_finais(pontos, vitorias, gols_pro, gols_contra, rng)

    return {"times": times, "posicao": posicao, "pontos": pontos}


# =============================================================================
# 4. RESUMO DOS RESULTADOS
# =============================================================================

def resumir(sim: dict) -> pd.DataFrame:
    times, posicao, pontos, n = sim["times"], sim["posicao"], sim["pontos"], sim["n"]
    n_times = len(times)
    caiu = posicao > n_times - N_REBAIXADOS

    # "Hoje" = arquivo do football-data + resultados manuais já conhecidos
    jogos = sim["jogos"]
    conhecidos = jogos[jogos["conhecido"]]
    pontos_hoje = sim["tabela"]["pontos"].copy()
    for _, j in conhecidos.iterrows():
        gm, gv = j["gols_mandante"], j["gols_visitante"]
        pontos_hoje[j["mandante"]] += 3 if gm > gv else (1 if gm == gv else 0)
        pontos_hoje[j["visitante"]] += 3 if gv > gm else (1 if gm == gv else 0)
    a_jogar = jogos[~jogos["conhecido"]]

    resumo = pd.DataFrame({
        "pontos_hoje": pontos_hoje.loc[times].to_numpy(),
        "jogos_restantes": [((a_jogar["mandante"] == t) | (a_jogar["visitante"] == t)).sum() for t in times],
        "pontos_finais_medio": pontos.mean(axis=0),
        "pontos_p10": np.percentile(pontos, 10, axis=0),
        "pontos_p90": np.percentile(pontos, 90, axis=0),
        "posicao_media": posicao.mean(axis=0),
        "prob_rebaixamento": caiu.mean(axis=0),
        "prob_titulo": (posicao == 1).mean(axis=0),
    }, index=pd.Index(times, name="time"))
    # Erro do sorteio (Monte Carlo): quanto a probabilidade oscilaria se
    # rodássemos com outra semente. Diminui com mais simulações.
    p = resumo["prob_rebaixamento"]
    resumo["erro_monte_carlo"] = np.sqrt(p * (1 - p) / n)
    return resumo.sort_values("prob_rebaixamento", ascending=False)


def pontos_de_corte(sim: dict) -> pd.Series:
    """Pontos do 17º colocado (o primeiro a cair) em cada simulação.
    Responde: 'quantos pontos salvam em 2026?'"""
    n_times = len(sim["times"])
    primeiro_a_cair = sim["posicao"] == n_times - N_REBAIXADOS + 1
    return pd.Series(sim["pontos"][primeiro_a_cair])


def quartetos_mais_comuns(sim: dict, top: int = 5) -> list:
    n_times = len(sim["times"])
    contagem = Counter()
    for linha in sim["posicao"]:
        caidos = tuple(sorted(np.array(sim["times"])[linha > n_times - N_REBAIXADOS]))
        contagem[caidos] += 1
    return [(q, c / sim["n"]) for q, c in contagem.most_common(top)]


def checar(sim: dict) -> None:
    """Testes de sanidade da simulação."""
    jogos, tabela = sim["jogos"], sim["tabela"]
    for t in sim["times"]:
        total = tabela.loc[t, "jogos"] + ((jogos["mandante"] == t) | (jogos["visitante"] == t)).sum()
        assert total == 38, f"{t} terminaria com {total} jogos"
    assert (np.sort(sim["posicao"], axis=1) == np.arange(1, len(sim["times"]) + 1)).all()
    print("  Cada time termina com 38 jogos e cada simulação tem posições 1–20 sem repetição: OK")


# =============================================================================
# 5. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulação Monte Carlo do Brasileirão 2026.")
    parser.add_argument("--n", type=int, default=20000, help="número de simulações (padrão 20.000)")
    parser.add_argument("--estatico", action="store_true",
                        help="não atualiza o Elo durante a simulação (só para comparação)")
    parser.add_argument("--incerteza-elo", type=float, default=INCERTEZA_PADRAO,
                        help=f"desvio padrão do sorteio somado ao Elo inicial (padrão {INCERTEZA_PADRAO:g})")
    args = parser.parse_args()

    n_formatado = f"{args.n:,}".replace(",", ".")   # 20000 -> 20.000
    print(f"=== Simulando o Brasileirão {TEMPORADA} ({n_formatado} vezes, "
          f"{'Elo congelado' if args.estatico else 'Elo dinâmico'}, "
          f"incerteza do Elo = {args.incerteza_elo:g}) ===")
    sim = simular(args.n, estatico=args.estatico, incerteza_elo=args.incerteza_elo)
    jogos = sim["jogos"]
    print(f"  {len(jogos)} jogos a definir ({jogos['conhecido'].sum()} com resultado já conhecido)")
    checar(sim)

    resumo = resumir(sim)
    print("\n=== Probabilidade de rebaixamento ===")
    tabela = pd.DataFrame({
        "pts hoje": resumo["pontos_hoje"].astype(int),
        "jogos rest.": resumo["jogos_restantes"],
        "pts finais (média)": resumo["pontos_finais_medio"].round(1),
        "faixa 80%": [f"{a:.0f}–{b:.0f}" for a, b in zip(resumo["pontos_p10"], resumo["pontos_p90"])],
        "rebaixamento": (resumo["prob_rebaixamento"] * 100).round(1).astype(str) + "%",
    })
    print(tabela.head(10).to_string())
    erro_max = resumo["erro_monte_carlo"].max() * 100
    print(f"\n  Erro do sorteio: no máximo ±{erro_max:.1f} p.p. (1 erro padrão)")

    corte = pontos_de_corte(sim)
    print("\n=== Quantos pontos salvam? (pontos do 17º colocado) ===")
    print(f"  mediana {corte.median():.0f} pontos; em 80% das simulações, entre "
          f"{corte.quantile(.1):.0f} e {corte.quantile(.9):.0f}")
    for pts in range(int(corte.quantile(.05)), int(corte.quantile(.95)) + 2):
        print(f"  com {pts} pontos, o time se salva em {(corte < pts).mean():.0%} das simulações"
              f" (sem contar desempates)")

    print("\n=== Quartetos rebaixados mais prováveis ===")
    for quarteto, prob in quartetos_mais_comuns(sim):
        print(f"  {prob:5.1%}  {', '.join(quarteto)}")

    print("\n=== Salvando ===")
    sufixo = "_estatico" if args.estatico else ""
    resumo.to_csv(PROCESSADOS / f"simulacao_{TEMPORADA}{sufixo}.csv")
    dist = pd.DataFrame({t: np.bincount(sim["posicao"][:, i], minlength=21)[1:] / sim["n"]
                         for i, t in enumerate(sim["times"])}, index=pd.Index(range(1, 21), name="posicao"))
    dist.to_csv(PROCESSADOS / f"simulacao_posicoes_{TEMPORADA}{sufixo}.csv")
    # Resumo extra para a página (atualizar.py): linha de corte e Z4 mais prováveis
    extras = {
        "n": int(sim["n"]),
        "corte": {"p10": float(corte.quantile(.1)), "mediana": float(corte.median()),
                  "p90": float(corte.quantile(.9))},
        # P(salvar) com X pontos, sem contar desempates
        "salva_com": {int(p): round(float((corte < p).mean()), 4)
                      for p in range(int(corte.quantile(.02)), int(corte.quantile(.98)) + 3)},
        "quartetos": [[list(q), round(float(p), 4)] for q, p in quartetos_mais_comuns(sim)],
    }
    (PROCESSADOS / f"simulacao_extras_{TEMPORADA}{sufixo}.json").write_text(
        json.dumps(extras, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  arquivos salvos em {PROCESSADOS}")


if __name__ == "__main__":
    main()
