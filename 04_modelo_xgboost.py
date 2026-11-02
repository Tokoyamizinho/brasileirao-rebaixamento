"""
04_modelo_xgboost.py
====================

Etapa 2 da metodologia: XGBOOST.

A PERGUNTA
----------
A regressão logística (etapa 1) só enxerga relações "em linha reta": cada 100
pontos de Elo a mais aumentam a chance de vitória sempre do mesmo jeito, e
cada feature age sozinha. O XGBoost é mais flexível:
  - relações não lineares (ex.: o efeito da diferença de Elo pode "saturar");
  - interações (ex.: a forma recente pode importar só para times fracos, ou só
    na reta final).
Se essas coisas existirem nos dados, o XGBoost deve errar menos.

COMO O XGBOOST FUNCIONA (resumo)
--------------------------------
Ele soma centenas de árvores de decisão pequenas. Cada árvore nova tenta
corrigir os erros das anteriores ("boosting"). Isso dá muito poder, e também
facilidade para DECORAR os dados de treino (overfitting). Os hiperparâmetros
controlam esse equilíbrio:
  max_depth          tamanho de cada árvore (1 = só uma pergunta por árvore)
  n_estimators       quantas árvores
  learning_rate      quanto cada árvore corrige (menor = mais cauteloso)
  min_child_weight   mínimo de dados em cada "folha" (maior = mais cauteloso)
  subsample / colsample_bytree   fração de jogos / features que cada árvore vê
  reg_lambda         penalidade para valores extremos nas folhas

AVALIAÇÃO JUSTA (o ponto mais importante deste script)
------------------------------------------------------
1. AJUSTE dos hiperparâmetros com Optuna, usando SÓ as temporadas 2017–2021
   (walk-forward: para cada S, treina com < S e mede o erro em S).
2. TESTE em 2022–2026, temporadas que o Optuna nunca viu, com os
   hiperparâmetros já congelados. A regressão logística é avaliada exatamente
   nos mesmos jogos.
Sem essa separação, o XGBoost "ganharia" só por ter sido ajustado nos dados
de avaliação (vimos isso acontecer com o Elo na prática 2).

Testamos dois conjuntos de features:
  so_elo: só a diferença de Elo         -> não linearidade ajuda?
  todas:  todas as features do 02       -> interações ajudam?

COMO RODAR
----------
    python -m pip install xgboost optuna
    python 04_modelo_xgboost.py                  # 100 tentativas do Optuna por conjunto (~5 min)
    python 04_modelo_xgboost.py --tentativas 30  # mais rápido

Lê:   data/processed/features_partidas.csv
Gera: data/processed/resultados_xgboost.csv   comparação no período de teste
      data/processed/params_xgboost.json      hiperparâmetros escolhidos
"""

from __future__ import annotations

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# Reaproveitamos dados, métricas e a regra de decisão da etapa 1, para a
# comparação ser exatamente nas mesmas condições.
logistica = import_module("03_modelo_logistico")

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"

TEMPORADAS_AJUSTE = list(range(2017, 2022))   # só para escolher hiperparâmetros
TEMPORADAS_TESTE = list(range(2022, 2027))    # avaliação final (nunca vistas pelo Optuna)
CLASSES = logistica.CLASSES                   # ["M", "E", "V"]
ROTULOS = {c: i for i, c in enumerate(CLASSES)}   # o XGBoost quer números: M=0, E=1, V=2

LADOS = ["mandante", "visitante"]
CONJUNTOS = {
    "so_elo": ["elo_diff"],
    "todas": (["elo_diff"]
              + [f"elo_pre_{l}" for l in LADOS]
              + [f"rodagem_{l}" for l in LADOS] + [f"promovido_{l}" for l in LADOS]
              + [f"{c}_5_{l}" for c in ("pontos", "gols_marcados", "gols_sofridos") for l in LADOS]
              + [f"jogos_janela_{l}" for l in LADOS]
              + [f"folga_z4_{l}" for l in LADOS] + [f"jogos_restantes_{l}" for l in LADOS]),
}


# =============================================================================
# 1. MODELO E VALIDAÇÃO
# =============================================================================

def novo_xgb(params: dict):
    from xgboost import XGBClassifier
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                         tree_method="hist", random_state=42, n_jobs=-1, **params)


def walk_forward_xgb(f: pd.DataFrame, colunas: list[str], params: dict,
                     temporadas: list[int], meia_vida) -> pd.DataFrame:
    """Para cada temporada S: treina com as anteriores, prevê S."""
    previsoes = []
    for temporada in temporadas:
        treino = f[f["temporada"] < temporada]
        teste = f[f["temporada"] == temporada]
        if teste.empty:
            continue
        pesos = logistica.pesos_por_temporada(treino["temporada"], temporada, meia_vida)
        modelo = novo_xgb(params)
        modelo.fit(treino[colunas], treino["resultado"].map(ROTULOS), sample_weight=pesos)
        previsoes.append(pd.DataFrame(modelo.predict_proba(teste[colunas]), columns=CLASSES, index=teste.index))
    return pd.concat(previsoes)


def otimizar(f: pd.DataFrame, colunas: list[str], tentativas: int, meia_vida) -> tuple[dict, float]:
    """Optuna escolhe os hiperparâmetros olhando SÓ para 2017–2021."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        }
        prob = walk_forward_xgb(f, colunas, params, TEMPORADAS_AJUSTE, meia_vida)
        return logistica.erros_por_jogo(f, prob)["log_loss"].mean()

    estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=tentativas)
    return estudo.best_params, estudo.best_value


# =============================================================================
# 2. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 2: XGBoost.")
    parser.add_argument("--tentativas", type=int, default=100, help="tentativas do Optuna por conjunto")
    args = parser.parse_args()

    f = logistica.carregar_features()
    meia = logistica.MEIA_VIDA_FINAL     # mesmo peso por recência da etapa 1
    print(f"=== XGBoost: ajuste em {TEMPORADAS_AJUSTE[0]}–{TEMPORADAS_AJUSTE[-1]}, "
          f"teste em {TEMPORADAS_TESTE[0]}–{TEMPORADAS_TESTE[-1]} ===")

    # --- Referências no período de teste ---
    teste_idx = f[f["temporada"].isin(TEMPORADAS_TESTE)].index
    prob_log = logistica.walk_forward(f, ["elo_diff"], meia).loc[lambda p: p.index.isin(teste_idx)]
    erros_log = logistica.erros_por_jogo(f, prob_log)
    mercado = f.loc[teste_idx, ["prob_mercado_mandante", "prob_mercado_empate", "prob_mercado_visitante"]]
    mercado.columns = CLASSES
    erros_mercado = logistica.erros_por_jogo(f, mercado)

    linhas = [
        {"modelo": "regressão logística (etapa 1)", "log_loss": erros_log["log_loss"].mean(),
         "brier": erros_log["brier"].mean()},
        {"modelo": "odds do mercado", "log_loss": erros_mercado["log_loss"].mean(),
         "brier": erros_mercado["brier"].mean()},
    ]
    params_salvos, probs = {}, {}
    for nome, colunas in CONJUNTOS.items():
        inicio = time.time()
        print(f"\n  Otimizando XGBoost '{nome}' ({len(colunas)} features, {args.tentativas} tentativas)...")
        params, erro_ajuste = otimizar(f, colunas, args.tentativas, meia)
        print(f"    pronto em {time.time() - inicio:.0f} s; log loss no AJUSTE {erro_ajuste:.4f}")
        print("    " + ", ".join(f"{k}={v:.3g}" for k, v in params.items()))

        probs[nome] = walk_forward_xgb(f, colunas, params, TEMPORADAS_TESTE, meia)
        erros = logistica.erros_por_jogo(f, probs[nome])
        cmp = logistica.comparar(erros, erros_log)
        linhas.append({"modelo": f"XGBoost {nome}", "log_loss": erros["log_loss"].mean(),
                       "brier": erros["brier"].mean(), **cmp,
                       "aprovado": logistica.aprovado(cmp)})
        params_salvos[nome] = params

    resultados = pd.DataFrame(linhas)
    print(f"\n=== Resultado no TESTE ({TEMPORADAS_TESTE[0]}–{TEMPORADAS_TESTE[-1]}, menor = melhor) ===")
    for _, r in resultados.iterrows():
        texto = f"  {r['modelo']:<32} log loss {r['log_loss']:.4f}   brier {r['brier']:.4f}"
        if pd.notna(r.get("dif_vs_base")):
            texto += (f"   vs logística {r['dif_vs_base']:+.4f} ± {r['erro_padrao']:.4f}"
                      f"   melhor em {int(r['temporadas_melhores'])}/{int(r['temporadas_total'])} temporadas")
        print(texto)

    aprovados = resultados[resultados["aprovado"].fillna(False).astype(bool)]
    print("\n=== Decisão ===")
    if aprovados.empty:
        print("  Nenhuma versão do XGBoost superou a regressão logística pela nossa regra "
              "(melhora > 2 erros padrão e em pelo menos metade das temporadas).")
    else:
        print(f"  Aprovado(s): {', '.join(aprovados['modelo'])}")

    # --- Viés e importância, para entender o que o XGBoost aprendeu ---
    print("\n=== Viés médio no teste (previsto − real) ===")
    for nome, p in [("logística", prob_log), *probs.items()]:
        r = f.loc[p.index, "resultado"]
        print(f"  {nome:<10} " + "   ".join(f"{c}: {p[c].mean() - (r == c).mean():+.3f}" for c in CLASSES))

    modelo_todas = novo_xgb(params_salvos["todas"])
    treino = f[f["temporada"] < TEMPORADAS_TESTE[-1]]
    modelo_todas.fit(treino[CONJUNTOS["todas"]], treino["resultado"].map(ROTULOS),
                     sample_weight=logistica.pesos_por_temporada(treino["temporada"], TEMPORADAS_TESTE[-1], meia))
    importancia = pd.Series(modelo_todas.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    # Cuidado ao ler: "importância" mostra o que o modelo USOU, não o que AJUDOU.
    # Uma feature pode ser muito usada e mesmo assim não melhorar as previsões
    # fora da amostra (o modelo pode estar ajustando ruído com ela).
    print("\n=== Importância das features no XGBoost 'todas' (% do ganho total) ===")
    print((importancia / importancia.sum() * 100).round(1).head(10).to_string())

    resultados.to_csv(PROCESSADOS / "resultados_xgboost.csv", index=False)
    (PROCESSADOS / "params_xgboost.json").write_text(json.dumps(params_salvos, indent=2), encoding="utf-8")
    print(f"\n  arquivos salvos em {PROCESSADOS}")


if __name__ == "__main__":
    main()
