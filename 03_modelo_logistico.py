"""
03_modelo_logistico.py
======================

Etapa 1 da metodologia: REGRESSÃO LOGÍSTICA MULTINOMIAL.

O QUE ESTE SCRIPT FAZ
---------------------
1. Testa, de forma incremental, quais features ajudam a prever o resultado
   (M = mandante vence, E = empate, V = visitante vence):
      modelo base = só a diferença de Elo
      base + rodagem, base + forma, ... (uma ideia de cada vez)
2. Testa se dar mais peso às temporadas recentes melhora o modelo.
3. Compara tudo com duas referências: um "chute" pelas frequências históricas
   e as odds do mercado.
4. Treina o modelo final com as features aprovadas e salva para a simulação.
5. Mostra as probabilidades para os próximos jogos do calendário.

COMO A AVALIAÇÃO FUNCIONA (validação "walk-forward")
----------------------------------------------------
Para cada temporada de teste S (2017 a 2026):
    treina com as temporadas 2013 ... S−1   ->   prevê todos os jogos de S
Ou seja, o modelo nunca vê o futuro, exatamente como vai acontecer em 2026.
2012 fica de fora do treino porque as notas do Elo ainda estão "aquecendo".

AS MÉTRICAS (quanto MENOR, melhor)
----------------------------------
log loss: para cada jogo, −log(probabilidade que o modelo deu ao que aconteceu).
    Dar 60% ao resultado certo custa 0,51; dar 10% custa 2,30. Pune muito
    quem tem certeza e erra. É a métrica principal.
Brier: soma dos erros ao quadrado nas três probabilidades. Mais "tolerante".

COMO DECIDIMOS SE UMA FEATURE FICA
----------------------------------
Uma feature entra no modelo final só se:
    (a) melhorar o log loss médio em pelo menos 2 erros padrão, e
    (b) melhorar em pelo menos metade das temporadas de teste.
O erro padrão mede quanto a diferença oscilaria só por acaso (é a mesma
ideia dos ±5 pontos da prática 1). Sem essa regra, escolheríamos ruído.

LIMITAÇÃO CONHECIDA
-------------------
Os hiperparâmetros do Elo foram escolhidos com 2014–2021, que se sobrepõe a
parte dos anos de teste. Isso deixa o Elo um pouco otimista, mas afeta todas as
variantes por igual, então a comparação entre elas continua justa. O backtest
(07) fará a separação completa.

COMO RODAR
----------
    python 03_modelo_logistico.py

Lê:   data/processed/features_partidas.csv, estado_atual_2026.csv,
      data/processed/jogos_restantes_2026.csv, data/manual/calendario_2026.csv
Gera: data/processed/resultados_logistica.csv   tabela dos experimentos
      data/processed/modelo_logistico.joblib    modelo final
      data/processed/coeficientes_logistica.csv coeficientes, para interpretar
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"
CALENDARIO = PASTA / "data" / "manual" / "calendario_2026.csv"

PRIMEIRA_TEMPORADA_TREINO = 2013
TEMPORADAS_TESTE = list(range(2017, 2027))   # inclui 2026 (só os jogos já disputados)
CLASSES = ["M", "E", "V"]                    # ordem fixa das probabilidades

# Grupos de features. Usamos DIFERENÇAS (mandante − visitante): com só ~5 mil
# jogos, menos colunas = coeficientes mais estáveis. (Testamos também as versões
# com colunas separadas para mandante e visitante: foram iguais ou piores.)
BASE = ["elo_diff"]
GRUPOS = {
    "rodagem": ["rodagem_diff", "promovido_diff"],
    "pontos_5": ["pontos_5_diff"],
    "saldo_5": ["saldo_5_diff"],
    "gols_sofridos_5": ["gols_sofridos_5_diff"],
    "motivacao": ["na_briga_diff", "desenganado_diff"],
}

# Alternativas que SUBSTITUEM a base (em vez de se somar a ela).
# Elo do mercado: a nota de cada time corrigida pelas odds (ver 02_features.py).
ALTERNATIVAS_BASE = {
    "elo_mercado": ["elo_mercado_diff"],
    "elo + elo_mercado": ["elo_diff", "elo_mercado_diff"],
}

# Meia-vida do peso das temporadas: com 3, uma temporada de 3 anos atrás vale
# metade da mais recente. None = todas valem igual.
MEIAS_VIDA = [None, 3, 6]

# Decisão manual sobre o peso do MODELO FINAL (sobrepõe a regra automática).
# None = segue a regra. Escolhemos 6 mesmo sem melhora mensurável no log loss,
# porque três evidências independentes mostram que o fator casa caiu
# (práticas 1 e 2 e a calibração deste script), e o peso reduz o viés do
# mandante de ~2,4 para ~1,7 p.p. a custo praticamente zero.
MEIA_VIDA_FINAL = 6

# Regra de aprovação de uma feature (ver docstring)
MINIMO_ERROS_PADRAO = 2.0
MINIMO_FRACAO_TEMPORADAS = 0.5


# =============================================================================
# 2. DADOS
# =============================================================================

def carregar_features() -> pd.DataFrame:
    f = pd.read_csv(PROCESSADOS / "features_partidas.csv", parse_dates=["data"])
    f = f[f["temporada"] >= PRIMEIRA_TEMPORADA_TREINO].copy()
    acrescentar_diferencas(f)
    return f


def acrescentar_diferencas(f: pd.DataFrame) -> None:
    """Cria as colunas de diferença (mandante − visitante) usadas nos grupos."""
    f["rodagem_diff"] = f["rodagem_mandante"] - f["rodagem_visitante"]
    f["promovido_diff"] = f["promovido_mandante"] - f["promovido_visitante"]
    f["pontos_5_diff"] = f["pontos_5_mandante"] - f["pontos_5_visitante"]
    saldo_m = f["gols_marcados_5_mandante"] - f["gols_sofridos_5_mandante"]
    saldo_v = f["gols_marcados_5_visitante"] - f["gols_sofridos_5_visitante"]
    f["saldo_5_diff"] = saldo_m - saldo_v
    f["gols_sofridos_5_diff"] = f["gols_sofridos_5_mandante"] - f["gols_sofridos_5_visitante"]
    f["na_briga_diff"] = f["na_briga_mandante"] - f["na_briga_visitante"]
    f["desenganado_diff"] = f["desenganado_mandante"] - f["desenganado_visitante"]


# =============================================================================
# 3. MODELO E MÉTRICAS
# =============================================================================

def novo_modelo() -> object:
    """StandardScaler deixa todas as features na mesma escala (média 0, desvio 1).
    Assim os coeficientes ficam comparáveis entre si e a regularização padrão
    da LogisticRegression trata todas por igual."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def pesos_por_temporada(temporadas: pd.Series, temporada_teste: int, meia_vida: float | None):
    if meia_vida is None:
        return None
    idade = (temporada_teste - 1) - temporadas      # 0 para a temporada mais recente
    return 0.5 ** (idade / meia_vida)


def prever(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """predict_proba devolve as colunas na ordem de modelo.classes_ (alfabética:
    E, M, V). Reordenamos para M, E, V, que é mais natural de ler."""
    p = pd.DataFrame(modelo.predict_proba(X), columns=modelo.classes_, index=X.index)
    return p[CLASSES]


def walk_forward(f: pd.DataFrame, colunas: list[str], meia_vida: float | None = None) -> pd.DataFrame:
    """Probabilidades fora da amostra para todos os jogos das temporadas de teste."""
    previsoes = []
    for temporada in TEMPORADAS_TESTE:
        treino = f[f["temporada"] < temporada]
        teste = f[f["temporada"] == temporada]
        if teste.empty:
            continue
        modelo = novo_modelo()
        modelo.fit(treino[colunas], treino["resultado"],
                   logisticregression__sample_weight=pesos_por_temporada(treino["temporada"], temporada, meia_vida))
        previsoes.append(prever(modelo, teste[colunas]))
    return pd.concat(previsoes)


def erros_por_jogo(f: pd.DataFrame, prob: pd.DataFrame) -> pd.DataFrame:
    """Log loss e Brier de cada jogo (depois tiramos médias)."""
    real = f.loc[prob.index, "resultado"]
    indice_real = real.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p = prob[CLASSES].to_numpy()
    acertou = np.zeros_like(p)
    acertou[np.arange(len(p)), indice_real] = 1
    return pd.DataFrame({
        "temporada": f.loc[prob.index, "temporada"],
        "log_loss": -np.log(np.clip(p[np.arange(len(p)), indice_real], 1e-15, 1)),
        "brier": ((p - acertou) ** 2).sum(axis=1),
    }, index=prob.index)


def comparar(erros: pd.DataFrame, erros_base: pd.DataFrame) -> dict:
    """Compara uma variante com o modelo base, JOGO A JOGO (comparação pareada):
    como os dois modelos preveem exatamente os mesmos jogos, olhamos a diferença
    em cada jogo. O erro padrão dessa diferença diz quanto ela oscilaria por acaso."""
    dif = erros["log_loss"] - erros_base["log_loss"]
    erro_padrao = dif.std() / np.sqrt(len(dif))
    por_temporada = (erros.groupby("temporada")["log_loss"].mean()
                     - erros_base.groupby("temporada")["log_loss"].mean())
    return {
        "dif_vs_base": dif.mean(),
        "erro_padrao": erro_padrao,
        "temporadas_melhores": int((por_temporada < 0).sum()),
        "temporadas_total": len(por_temporada),
    }


# =============================================================================
# 4. EXPERIMENTOS
# =============================================================================

def rodar_experimentos(f: pd.DataFrame) -> tuple[pd.DataFrame, float | None, list[str]]:
    linhas = []

    def registrar(nome, tipo, erros, cmp=None):
        linha = {"modelo": nome, "tipo": tipo,
                 "log_loss": erros["log_loss"].mean(), "brier": erros["brier"].mean()}
        if cmp:
            linha.update(cmp)
        linhas.append(linha)

    # --- Referências ---
    # "Chute": em toda temporada de teste, usa as frequências de M/E/V do treino
    chute = []
    for temporada in TEMPORADAS_TESTE:
        freq = f.loc[f["temporada"] < temporada, "resultado"].value_counts(normalize=True)
        teste = f[f["temporada"] == temporada]
        chute.append(pd.DataFrame([freq[CLASSES].to_numpy()] * len(teste), columns=CLASSES, index=teste.index))
    registrar("chute (frequências históricas)", "referência", erros_por_jogo(f, pd.concat(chute)))

    teste_idx = f[f["temporada"].isin(TEMPORADAS_TESTE)].index
    mercado = f.loc[teste_idx, ["prob_mercado_mandante", "prob_mercado_empate", "prob_mercado_visitante"]]
    mercado.columns = CLASSES
    registrar("odds do mercado", "referência", erros_por_jogo(f, mercado))

    # --- Peso das temporadas (com o modelo base) ---
    erros_meia = {}
    for meia in MEIAS_VIDA:
        erros_meia[meia] = erros_por_jogo(f, walk_forward(f, BASE, meia))
    for meia in MEIAS_VIDA:
        nome = "base, todas as temporadas com o mesmo peso" if meia is None else f"base, meia-vida de {meia} temporadas"
        cmp = None if meia is None else comparar(erros_meia[meia], erros_meia[None])
        registrar(nome, "peso das temporadas", erros_meia[meia], cmp)
    # Só adotamos o peso por recência se ele passar na mesma regra das features
    meia_escolhida = None
    for meia in MEIAS_VIDA[1:]:
        if aprovado(comparar(erros_meia[meia], erros_meia[None])):
            if meia_escolhida is None or erros_meia[meia]["log_loss"].mean() < erros_meia[meia_escolhida]["log_loss"].mean():
                meia_escolhida = meia

    # --- Features, uma de cada vez ---
    erros_base = erros_meia[meia_escolhida]
    registrar("BASE: só diferença de Elo", "features", erros_base)
    aprovadas = []
    for grupo, colunas in GRUPOS.items():
        erros = erros_por_jogo(f, walk_forward(f, BASE + colunas, meia_escolhida))
        cmp = comparar(erros, erros_base)
        registrar(f"base + {grupo}", "features", erros, cmp)
        if aprovado(cmp):
            aprovadas.append(grupo)

    todas = BASE + [c for cols in GRUPOS.values() for c in cols]
    erros = erros_por_jogo(f, walk_forward(f, todas, meia_escolhida))
    registrar("base + todas", "features", erros, comparar(erros, erros_base))

    colunas_finais = BASE + [c for g in aprovadas for c in GRUPOS[g]]

    # --- Alternativas à base: se alguma passar na regra, a melhor vira o
    # modelo final (os grupos acima foram testados sobre o Elo comum) ---
    melhor_alt, melhor_ll = None, erros_base["log_loss"].mean()
    for nome, colunas in ALTERNATIVAS_BASE.items():
        erros = erros_por_jogo(f, walk_forward(f, colunas, meia_escolhida))
        cmp = comparar(erros, erros_base)
        registrar(nome, "alternativas à base", erros, cmp)
        if aprovado(cmp) and erros["log_loss"].mean() < melhor_ll:
            melhor_alt, melhor_ll = nome, erros["log_loss"].mean()
    if melhor_alt is not None:
        colunas_finais = ALTERNATIVAS_BASE[melhor_alt]
    return pd.DataFrame(linhas), meia_escolhida, colunas_finais


def aprovado(cmp: dict) -> bool:
    melhora_clara = cmp["dif_vs_base"] < -MINIMO_ERROS_PADRAO * cmp["erro_padrao"]
    consistente = cmp["temporadas_melhores"] >= MINIMO_FRACAO_TEMPORADAS * cmp["temporadas_total"]
    return bool(melhora_clara and consistente)


def imprimir_resultados(resultados: pd.DataFrame) -> None:
    for tipo, bloco in resultados.groupby("tipo", sort=False):
        print(f"\n  [{tipo}]")
        for _, r in bloco.iterrows():
            texto = f"    {r['modelo']:<45} log loss {r['log_loss']:.4f}   brier {r['brier']:.4f}"
            if pd.notna(r.get("dif_vs_base")):
                texto += (f"   dif {r['dif_vs_base']:+.4f} ± {r['erro_padrao']:.4f}"
                          f"   melhor em {int(r['temporadas_melhores'])}/{int(r['temporadas_total'])} temporadas")
            print(texto)


# =============================================================================
# 5. MODELO FINAL
# =============================================================================

def treinar_final(f: pd.DataFrame, colunas: list[str], meia_vida: float | None):
    """Treina com TODOS os jogos disputados (inclusive os de 2026)."""
    ultima = f["temporada"].max()
    pesos = pesos_por_temporada(f["temporada"], ultima + 1, meia_vida)
    modelo = novo_modelo()
    modelo.fit(f[colunas], f["resultado"], logisticregression__sample_weight=pesos)
    return modelo


def coeficientes(modelo, colunas: list[str]) -> pd.DataFrame:
    """Como as features foram padronizadas, cada coeficiente diz quanto aquela
    classe fica mais (positivo) ou menos (negativo) provável quando a feature
    aumenta 1 desvio padrão."""
    reg = modelo[-1]
    tabela = pd.DataFrame(reg.coef_, index=reg.classes_, columns=colunas)
    tabela["intercepto"] = reg.intercept_
    return tabela.loc[CLASSES]


def calibracao(f: pd.DataFrame, prob: pd.DataFrame) -> pd.DataFrame:
    """Para a probabilidade de vitória do mandante: em cada faixa, a média prevista
    deve ficar perto da frequência real."""
    tabela = pd.DataFrame({"prevista": prob["M"],
                           "real": (f.loc[prob.index, "resultado"] == "M").astype(float)})
    tabela["faixa"] = pd.cut(tabela["prevista"], bins=[0, .3, .4, .5, .6, .7, 1])
    return tabela.groupby("faixa", observed=True).agg(
        jogos=("real", "size"), prevista=("prevista", "mean"), real=("real", "mean"))


def prever_proximos(modelo, colunas: list[str]) -> pd.DataFrame | None:
    """Probabilidades para os próximos jogos ainda não disputados no nosso arquivo,
    usando o estado ATUAL de cada time (Elo de hoje etc.).

    Só funciona para features que existem no estado atual. É uma previsão
    "estática": a simulação (06) vai atualizar o Elo a cada jogo sorteado."""
    if not CALENDARIO.exists():
        return None
    estado = pd.read_csv(PROCESSADOS / "estado_atual_2026.csv", index_col="time")
    restantes = pd.read_csv(PROCESSADOS / "jogos_restantes_2026.csv")
    calendario = pd.read_csv(CALENDARIO).merge(restantes, on=["mandante", "visitante"])
    proxima = calendario.dropna(subset=["data"]).sort_values("data")["rodada"].min()
    jogos = calendario[calendario["rodada"] == proxima].copy()

    x = pd.DataFrame(index=jogos.index)
    x["elo_diff"] = jogos["mandante"].map(estado["elo"]).to_numpy() - jogos["visitante"].map(estado["elo"]).to_numpy()
    if "elo_mercado" in estado.columns:
        x["elo_mercado_diff"] = (jogos["mandante"].map(estado["elo_mercado"]).to_numpy()
                                 - jogos["visitante"].map(estado["elo_mercado"]).to_numpy())
    if set(colunas) - set(x.columns):
        print("  (previsão dos próximos jogos só implementada para o modelo com Elo)")
        return None
    prob = prever(modelo, x[colunas])
    return jogos[["rodada", "data", "mandante", "visitante"]].join((prob * 100).round(0))


# =============================================================================
# 6. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== 1. Dados ===")
    f = carregar_features()
    print(f"  {len(f)} partidas ({f['temporada'].min()}–{f['temporada'].max()}); "
          f"teste walk-forward em {TEMPORADAS_TESTE[0]}–{TEMPORADAS_TESTE[-1]}")

    print("\n=== 2. Experimentos (menor = melhor) ===")
    resultados, meia, colunas = rodar_experimentos(f)
    imprimir_resultados(resultados)
    resultados.to_csv(PROCESSADOS / "resultados_logistica.csv", index=False)

    print("\n=== 3. Decisão ===")
    print(f"  Regra: melhora maior que {MINIMO_ERROS_PADRAO:.0f} erros padrão e em pelo menos "
          f"{MINIMO_FRACAO_TEMPORADAS:.0%} das temporadas.")
    print(f"  Peso por recência pela regra: {'nenhum' if meia is None else f'meia-vida de {meia} temporadas'}")
    if MEIA_VIDA_FINAL is not None:
        meia = MEIA_VIDA_FINAL
        print(f"  Decisão manual (MEIA_VIDA_FINAL): meia-vida de {meia} temporadas no modelo final")
    print(f"  Features do modelo final: {colunas}")

    print("\n=== 4. Calibração do modelo final (fora da amostra) ===")
    prob_teste = walk_forward(f, colunas, meia)
    print(calibracao(f, prob_teste).round(3).to_string())
    # Viés: a probabilidade média prevista para cada resultado deveria ficar
    # perto da frequência real. Um viés positivo em M = o modelo dá vantagem
    # de casa demais (o fator casa caiu ao longo dos anos, lembra?).
    real = f.loc[prob_teste.index, "resultado"]
    for classe in CLASSES:
        print(f"  {classe}: previsto {prob_teste[classe].mean():.3f}   real {(real == classe).mean():.3f}")

    modelo = treinar_final(f, colunas, meia)
    coef = coeficientes(modelo, colunas)
    print("\n=== 5. Coeficientes (features padronizadas) ===")
    print(coef.round(3).to_string())
    coef.to_csv(PROCESSADOS / "coeficientes_logistica.csv")
    joblib.dump({"modelo": modelo, "colunas": colunas, "meia_vida": meia},
                PROCESSADOS / "modelo_logistico.joblib")

    proximos = prever_proximos(modelo, colunas)
    if proximos is not None:
        print("\n=== 6. Próximos jogos no arquivo (probabilidades em %) ===")
        print(proximos.to_string(index=False))

    print(f"\n  arquivos salvos em {PROCESSADOS}")


if __name__ == "__main__":
    main()
