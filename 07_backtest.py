"""
07_backtest.py
==============

Backtest: o modelo teria acertado nas temporadas passadas?

A PERGUNTA
----------
Quando o modelo diz "60% de chance de cair", o time cai mesmo em ~60% das vezes?
Só dá para saber testando em temporadas cujo final já conhecemos.

COMO FUNCIONA
-------------
Para cada temporada S de 2015 a 2025 e para cada "corte" R (rodadas 10, 15, 20,
25, 30 e 34), fingimos estar no fim da rodada R de S:
  1. só os primeiros R x 10 jogos de S são conhecidos;
  2. o Elo é calculado só com os jogos até esse ponto;
  3. o modelo logístico é treinado só com temporadas anteriores a S;
  4. os placares sorteados vêm das 5 temporadas anteriores a S;
  5. simulamos o resto da temporada, na ordem real dos jogos;
  6. comparamos as probabilidades com quem caiu de verdade.
Nenhuma informação do futuro entra: é exatamente a situação de 2026.

"Rodada R" = os primeiros R x 10 jogos da temporada em ordem cronológica.
O arquivo não traz a rodada oficial; jogos adiados ficam naturalmente para depois.

AS MÉTRICAS
-----------
Brier (principal): média de (probabilidade − resultado)², em que resultado = 1 se
    caiu e 0 se não caiu. 0 é perfeito. Quem chuta 20% para todo mundo tira 0,16.
Calibração: agrupamos as previsões por faixa (ex.: 10–25%) e conferimos se a
    frequência real de queda caiu dentro da faixa.

COMPARAÇÕES
-----------
- Modelo com Elo dinâmico (o nosso padrão) e com Elo congelado.
- "Z4 atual": diz 100% para quem está no Z4 no corte e 0% para os demais.
- "Histórico por posição": a frequência com que o time naquela posição, naquela
  rodada, caiu nas OUTRAS temporadas. É o que muita gente faz "de cabeça".

LIMITAÇÃO
---------
Os hiperparâmetros do Elo foram escolhidos com 2014–2021 (ver 02_features.py).
Então 2015–2021 tem um leve "vazamento" e 2022–2025 é o teste realmente limpo.
O resumo mostra os dois períodos separados.

COMO RODAR
----------
    python 07_backtest.py                  # 5.000 simulações por cenário (~3–5 min)
    python 07_backtest.py --n 2000         # mais rápido, um pouco mais de ruído
    python 07_backtest.py --sem-estatico   # pula a versão com Elo congelado

    # Variações para testar ideias (cada uma salva arquivos com um sufixo próprio):
    python 07_backtest.py --sem-estatico --features elo_diff,na_briga_diff,desenganado_diff
    python 07_backtest.py --sem-estatico --incerteza-elo 0    # sem incerteza (padrão: 75)

Gera: data/processed/backtest_previsoes.csv   uma linha por (temporada, corte, time)
      data/processed/backtest_resumo.csv      métricas por corte e modelo
"""

from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Reaproveitamos o código dos outros scripts, para garantir que o backtest
# testa EXATAMENTE o que usamos em 2026.
features = import_module("02_features")
modelo_log = import_module("03_modelo_logistico")
simulacao = import_module("06_simulacao")

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"

TEMPORADAS = list(range(2015, 2026))
CORTES = [10, 15, 20, 25, 30, 34]
PERIODO_LIMPO = (2022, 2025)        # anos que não influenciaram os parâmetros do Elo
N_REBAIXADOS = 4


# =============================================================================
# 1. PREPARAÇÃO
# =============================================================================

def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    partidas = pd.read_csv(PROCESSADOS / "partidas.csv", parse_dates=["data"])
    partidas = partidas.sort_values(["temporada", "data", "id_partida"]).reset_index(drop=True)
    feats = pd.read_csv(PROCESSADOS / "features_partidas.csv")
    modelo_log.acrescentar_diferencas(feats)        # cria as colunas *_diff
    finais = pd.read_csv(PROCESSADOS / "tabelas_finais.csv")
    params = json.loads((PROCESSADOS / "params_elo.json").read_text(encoding="utf-8"))
    return partidas, feats, finais, params


def classificacao_no_corte(conhecidos: pd.DataFrame, times: list[str]) -> pd.DataFrame:
    """Tabela no momento do corte, incluindo todos os times da temporada
    (mesmo que algum ainda não tenha jogado), com a posição de cada um."""
    tabela = simulacao.tabela_atual(conhecidos).reindex(times, fill_value=0)
    tabela["saldo"] = tabela["gols_pro"] - tabela["gols_contra"]
    ordem = tabela.sort_values(["pontos", "vitorias", "saldo", "gols_pro"], ascending=False)
    tabela["posicao"] = pd.Series(range(1, len(times) + 1), index=ordem.index)
    return tabela


def treinar_modelo(feats: pd.DataFrame, temporada: int, colunas: list[str]):
    """Modelo logístico treinado só com temporadas ANTERIORES à testada."""
    treino = feats[feats["temporada"].between(modelo_log.PRIMEIRA_TEMPORADA_TREINO, temporada - 1)]
    meia = modelo_log.MEIA_VIDA_FINAL
    modelo = modelo_log.novo_modelo()
    modelo.fit(treino[colunas], treino["resultado"],
               logisticregression__sample_weight=modelo_log.pesos_por_temporada(treino["temporada"], temporada, meia))
    return modelo


# =============================================================================
# 2. UM CENÁRIO (temporada S, corte R)
# =============================================================================

def rodar_cenario(partidas, modelo, params, temporada, corte, n, estatico,
                  colunas=("elo_diff",), incerteza_elo=0.0, k_simulacao=None) -> pd.DataFrame:
    da_temporada = partidas[partidas["temporada"] == temporada]
    times = sorted(set(da_temporada["mandante"]) | set(da_temporada["visitante"]))
    n_conhecidos = corte * len(times) // 2           # 10 jogos por rodada
    conhecidos = da_temporada.iloc[:n_conhecidos]
    a_simular = da_temporada.iloc[n_conhecidos:][["mandante", "visitante", "data"]].copy()
    a_simular = a_simular.rename(columns={"data": "data_ordem"})
    a_simular["conhecido"] = False
    a_simular["gols_mandante"] = np.nan
    a_simular["gols_visitante"] = np.nan

    # Elo com tudo o que se sabia no corte: temporadas anteriores + jogos conhecidos
    historico = pd.concat([partidas[partidas["temporada"] < temporada], conhecidos])
    _, ratings = features.calcular_elo(historico, **params)
    params_mercado = features.carregar_params_mercado()
    ratings_mercado = None
    if "elo_mercado_diff" in colunas:
        _, ratings_mercado = features.calcular_elo(historico, alvo="mercado", **params_mercado)
        ratings_mercado = pd.Series(ratings_mercado)
    # K da nota do mercado nos jogos SIMULADOS (só a simulação usa esta chave)
    params_mercado = {**params_mercado,
                      "K_simulacao": k_simulacao if k_simulacao is not None else simulacao.K_SIMULACAO_MERCADO}

    tabela = classificacao_no_corte(conhecidos, times)
    placares = simulacao.distribuicao_placares(partidas, (temporada - 5, temporada - 1))
    rng = np.random.default_rng(temporada * 100 + corte)   # reprodutível

    sim = simulacao.simular_jogos(tabela[["jogos", "pontos", "vitorias", "gols_pro", "gols_contra"]],
                                  pd.Series(ratings), a_simular.reset_index(drop=True),
                                  modelo, params, placares, n, rng, estatico,
                                  colunas=list(colunas), incerteza_elo=incerteza_elo,
                                  elo_mercado_inicial=ratings_mercado, params_mercado=params_mercado)
    prob = (sim["posicao"] > len(times) - N_REBAIXADOS).mean(axis=0)
    return pd.DataFrame({
        "temporada": temporada, "corte": corte, "time": sim["times"],
        "posicao_no_corte": tabela.loc[sim["times"], "posicao"].to_numpy(),
        "pontos_no_corte": tabela.loc[sim["times"], "pontos"].to_numpy(),
        "prob": prob,
    })


# =============================================================================
# 3. LINHA DE BASE "HISTÓRICO POR POSIÇÃO"
# =============================================================================

def historico_por_posicao(partidas: pd.DataFrame, finais: pd.DataFrame) -> pd.DataFrame:
    """Para cada temporada encerrada, corte e time: posição no corte e se caiu."""
    linhas = []
    for temporada in sorted(finais["temporada"].unique()):
        da_temporada = partidas[partidas["temporada"] == temporada]
        times = sorted(set(da_temporada["mandante"]) | set(da_temporada["visitante"]))
        caiu = finais[finais["temporada"] == temporada].set_index("time")["rebaixado"]
        for corte in CORTES:
            tab = classificacao_no_corte(da_temporada.iloc[:corte * len(times) // 2], times)
            for time_, pos in tab["posicao"].items():
                linhas.append((temporada, corte, pos, caiu[time_]))
    return pd.DataFrame(linhas, columns=["temporada", "corte", "posicao", "caiu"])


def prob_historica(hist: pd.DataFrame, temporada: int, corte: int, posicao: int) -> float:
    """Frequência de queda na mesma posição e corte, nas OUTRAS temporadas."""
    outras = hist[(hist["temporada"] != temporada) & (hist["corte"] == corte) & (hist["posicao"] == posicao)]
    return outras["caiu"].mean()


# =============================================================================
# 4. MÉTRICAS
# =============================================================================

def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def resumo_metricas(prev: pd.DataFrame, modelos: list[str]) -> pd.DataFrame:
    linhas = []
    for periodo, filtro in [("2015–2025", prev["temporada"] >= 0),
                            (f"{PERIODO_LIMPO[0]}–{PERIODO_LIMPO[1]} (limpo)",
                             prev["temporada"].between(*PERIODO_LIMPO))]:
        for corte in CORTES + ["todos"]:
            sel = prev[filtro] if corte == "todos" else prev[filtro & (prev["corte"] == corte)]
            linha = {"periodo": periodo, "corte": corte}
            for m in modelos:
                linha[m] = brier(sel[m], sel["caiu"])
            linhas.append(linha)
    return pd.DataFrame(linhas)


def calibracao(prev: pd.DataFrame, coluna: str) -> pd.DataFrame:
    faixas = [0, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 1.0]
    t = prev.assign(faixa=pd.cut(prev[coluna], bins=faixas, include_lowest=True))
    return t.groupby("faixa", observed=True).agg(
        previsoes=("caiu", "size"), prob_media=(coluna, "mean"),
        frequencia_real=("caiu", "mean"), quedas=("caiu", "sum"))


# =============================================================================
# 5. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest das probabilidades de rebaixamento.")
    parser.add_argument("--n", type=int, default=5000, help="simulações por cenário (padrão 5.000)")
    parser.add_argument("--sem-estatico", action="store_true", help="não roda a versão com Elo congelado")
    # Padrão: as MESMAS features do modelo final escolhido pelo 03 (assim o
    # backtest testa exatamente o que usamos em 2026)
    caminho_modelo = PROCESSADOS / "modelo_logistico.joblib"
    padrao = ",".join(joblib.load(caminho_modelo)["colunas"]) if caminho_modelo.exists() else "elo_diff"
    parser.add_argument("--features", default=padrao,
                        help=f"features do modelo, separadas por vírgula (padrão: as do modelo final, {padrao})")
    parser.add_argument("--k-simulacao", type=float, default=None,
                        help="K da nota do mercado nos jogos simulados (padrão: o do params_elo_mercado.json ou o do Elo comum)")
    parser.add_argument("--incerteza-elo", type=float, default=simulacao.INCERTEZA_PADRAO,
                        help=f"desvio padrão do sorteio somado ao Elo inicial (padrão {simulacao.INCERTEZA_PADRAO:g})")
    args = parser.parse_args()
    colunas = args.features.split(",")

    # Variações salvam arquivos com sufixo, para não sobrescrever o backtest padrão
    sufixo = ""
    if args.features != padrao:
        sufixo += "_" + "+".join(c.replace("_diff", "") for c in colunas)
    if args.incerteza_elo != simulacao.INCERTEZA_PADRAO:
        sufixo += f"_incerteza{args.incerteza_elo:g}"
    if args.k_simulacao is not None:
        sufixo += f"_ksim{args.k_simulacao:g}"

    partidas, feats, finais, params = carregar()
    caiu = finais.set_index(["temporada", "time"])["rebaixado"]
    hist = historico_por_posicao(partidas, finais)

    print(f"=== Backtest: {len(TEMPORADAS)} temporadas x {len(CORTES)} cortes, "
          f"{args.n} simulações por cenário ===")
    print(f"  features: {colunas}   incerteza do Elo: {args.incerteza_elo:g}")
    inicio = time.time()
    resultados = []
    for temporada in TEMPORADAS:
        modelo = treinar_modelo(feats, temporada, colunas)
        for corte in CORTES:
            cen = rodar_cenario(partidas, modelo, params, temporada, corte, args.n, estatico=False,
                                colunas=colunas, incerteza_elo=args.incerteza_elo,
                                k_simulacao=args.k_simulacao)
            cen = cen.rename(columns={"prob": "modelo_dinamico"})
            if not args.sem_estatico:
                est = rodar_cenario(partidas, modelo, params, temporada, corte, args.n, estatico=True,
                                    colunas=colunas, incerteza_elo=args.incerteza_elo,
                                k_simulacao=args.k_simulacao)
                cen["modelo_estatico"] = est["prob"].to_numpy()
            resultados.append(cen)
        print(f"  {temporada} pronto ({time.time() - inicio:.0f} s)")

    prev = pd.concat(resultados, ignore_index=True)
    prev["caiu"] = [caiu[(t, tm)] for t, tm in zip(prev["temporada"], prev["time"])]
    prev["z4_atual"] = (prev["posicao_no_corte"] > 20 - N_REBAIXADOS).astype(float)
    prev["historico_posicao"] = [prob_historica(hist, t, c, p) for t, c, p in
                                 zip(prev["temporada"], prev["corte"], prev["posicao_no_corte"])]

    # Sanidade: em cada cenário, as probabilidades devem somar 4 (4 rebaixados)
    soma = prev.groupby(["temporada", "corte"])["modelo_dinamico"].sum()
    assert np.allclose(soma, N_REBAIXADOS), "probabilidades não somam 4 em algum cenário"
    print("  Probabilidades somam 4 em todos os cenários: OK")

    modelos = ["modelo_dinamico"] + ([] if args.sem_estatico else ["modelo_estatico"]) + \
              ["historico_posicao", "z4_atual"]
    resumo = resumo_metricas(prev, modelos)
    print("\n=== Brier por rodada do corte (menor = melhor; chutar 20% para todos dá 0,160) ===")
    for periodo, bloco in resumo.groupby("periodo", sort=False):
        print(f"\n  [{periodo}]")
        print(bloco.drop(columns="periodo").set_index("corte").round(4).to_string())

    print("\n=== Calibração do modelo dinâmico (todos os cenários) ===")
    print(calibracao(prev, "modelo_dinamico").round(3).to_string())

    print("\n=== Maiores surpresas na rodada 30: caíram com probabilidade baixa ===")
    r30 = prev[prev["corte"] == 30]
    print(r30[r30["caiu"] == 1].nsmallest(5, "modelo_dinamico")[
        ["temporada", "time", "posicao_no_corte", "pontos_no_corte", "modelo_dinamico"]].round(3).to_string(index=False))
    print("\n=== Salvações improváveis na rodada 30: escaparam com probabilidade alta ===")
    print(r30[r30["caiu"] == 0].nlargest(5, "modelo_dinamico")[
        ["temporada", "time", "posicao_no_corte", "pontos_no_corte", "modelo_dinamico"]].round(3).to_string(index=False))

    prev.to_csv(PROCESSADOS / f"backtest_previsoes{sufixo}.csv", index=False)
    resumo.to_csv(PROCESSADOS / f"backtest_resumo{sufixo}.csv", index=False)
    print(f"\n  arquivos salvos em {PROCESSADOS}  ({time.time() - inicio:.0f} s no total)")


if __name__ == "__main__":
    main()
