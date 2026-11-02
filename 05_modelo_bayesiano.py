"""
05_modelo_bayesiano.py
======================

Etapa 3 da metodologia: MODELO HIERÁRQUICO BAYESIANO DE GOLS.

A IDEIA
-------
Em vez de prever "vitória, empate ou derrota", este modelo prevê os GOLS.
Cada time tem duas forças: ATAQUE (quanto marca) e DEFESA (quanto sofre).
Os gols do mandante seguem uma distribuição de Poisson com média

    gols esperados do mandante  = exp(base + casa + ataque[mandante] + defesa[visitante])
    gols esperados do visitante = exp(base +        ataque[visitante] + defesa[mandante])

(defesa positiva = defesa RUIM, que sofre mais gols). É o modelo de Baio e
Blangiardo (2010), com uma extensão: as forças mudam de uma temporada para a
outra.

O QUE É "BAYESIANO" AQUI
------------------------
Não estimamos UM valor para cada força, e sim uma DISTRIBUIÇÃO de valores
plausíveis (a "posteriori"), combinando:
  - a priori: o que achávamos antes de ver os jogos (ex.: a força deste ano
    se parece com a do ano passado);
  - a verossimilhança: o que os placares mostram.
Quanto mais jogos um time faz, mais estreita fica a distribuição da força dele.

AS FORÇAS MUDAM ENTRE TEMPORADAS (processo AR(1))
-------------------------------------------------
    força[ano] = persistência x força[ano anterior] + mudança aleatória
A "persistência" é a versão bayesiana da regressão à média do Elo (prática 2):
quanto da força passa para o ano seguinte. Aqui ela NÃO é escolhida por nós:
o modelo a estima a partir dos dados, junto com o tamanho típico da mudança.
Times recém-promovidos recebem um ajuste próprio (efeito "promovido").

AS FORÇAS MUDAM TAMBÉM DENTRO DA TEMPORADA (--blocos, padrão 4)
---------------------------------------------------------------
Cada temporada é dividida em 4 blocos de ~95 jogos. De um bloco para o outro a
força dá um pequeno passo aleatório (tamanho "mudanca_intra", estimado). Assim
um time que melhora no 2º turno não fica preso à força média do ano.

FUSÃO COM O MERCADO (padrão; --sem-mercado desliga)
---------------------------------------------------
Os gols são uma medida ruidosa da força. As odds dos jogos PASSADOS são outra
medida, bem mais precisa. O modelo usa as duas ao mesmo tempo: a mesma força
(ataque − defesa) precisa explicar os placares E o que o mercado esperava
antes de cada jogo. Sem vazamento: só entram odds de jogos já disputados; os
jogos futuros são simulados apenas com as forças.

Resultado do backtest 2015–2025 (Brier, 4 cortes): versão só com gols 0,0797;
+ mercado 0,0736; + mercado + 4 blocos 0,0723; logística da etapa 1 0,0717.

POR QUE ISSO RESOLVE O EXCESSO DE CONFIANÇA?
--------------------------------------------
Na simulação, cada cenário usa um sorteio diferente da posteriori. Se não
sabemos bem a força do Grêmio, cada cenário usa um Grêmio um pouco diferente.
É o que fizemos "na mão" com --incerteza-elo 75, só que agora a incerteza vem
dos dados, e é maior para quem tem menos jogos.

COMO O MODELO É AJUSTADO (MCMC)
-------------------------------
A posteriori não tem fórmula fechada. O PyMC usa o algoritmo NUTS (MCMC) para
SORTEAR milhares de valores dela. Rodamos 4 "cadeias" independentes e
conferimos se chegaram ao mesmo lugar:
  r_hat perto de 1,00 (até 1,01) = as cadeias concordam
  divergências = 0               = o amostrador não teve problemas
  ess (amostra efetiva) > 400    = sorteios suficientes

INSTALAÇÃO
----------
    python -m pip install pymc nutpie
O nutpie é um amostrador mais rápido e não precisa de compilador C, o que
evita problemas no Windows. Sem ele, o script usa o amostrador padrão do PyMC.

COMO RODAR
----------
    python 05_modelo_bayesiano.py                       # ajusta e simula 2026 (~3 min)
    python 05_modelo_bayesiano.py --backtest --amostras 500   # backtest (~1 h)
    python 05_modelo_bayesiano.py --sem-mercado --blocos 1    # versão original, só gols

Lê:   data/processed/partidas.csv, data/raw/brasileirao_github_2003_2024.csv
      data/manual/calendario_2026.csv, resultados_recentes_2026.csv
Gera (o sufixo indica a versão; o padrão é _bayes_mercado_blocos4):
      data/processed/forcas_bayes_mercado_blocos4_2026.csv     ataque, defesa e força
      data/processed/simulacao_2026_bayes_mercado_blocos4.csv  probabilidades de rebaixamento
      data/processed/backtest_previsoes_bayes_mercado_blocos4.csv (com --backtest)
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

features = import_module("02_features")
simulacao = import_module("06_simulacao")

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"
TEMPORADA = 2026
N_REBAIXADOS = 4

JANELA = 4              # temporadas usadas em cada ajuste (a atual + 3 anteriores)
AMOSTRAS = 1000         # sorteios por cadeia, depois do aquecimento
AQUECIMENTO = 1000      # sorteios descartados enquanto o NUTS se ajusta
CADEIAS = 4
CORTES_BACKTEST = [10, 20, 30, 34]


# =============================================================================
# 1. DADOS
# =============================================================================

def conjunto_promovidos(partidas: pd.DataFrame) -> set:
    """(temporada, time) de quem estava na Série A naquele ano, mas não no anterior."""
    rodagem = features.calcular_rodagem(features.temporadas_na_serie_a(partidas))
    return set(zip(rodagem.loc[rodagem["promovido"] == 1, "temporada"],
                   rodagem.loc[rodagem["promovido"] == 1, "time"]))


def preparar(anteriores: pd.DataFrame, conhecidos: pd.DataFrame, temporada: int,
             times_temporada: list[str], promovidos: set, blocos: int = 1) -> dict:
    """Junta os jogos da janela e cria os índices que o modelo usa.

    anteriores: jogos das temporadas anteriores dentro da janela
    conhecidos: jogos já disputados na temporada que queremos prever
    blocos:     em quantos pedaços cada temporada é dividida (1 = força fixa no
                ano inteiro; 4 = a força pode mudar a cada ~10 rodadas)

    Cada (temporada, bloco) vira um "período". Na temporada atual, só entram
    os blocos que já têm jogos: a previsão usa a força do último período.
    """
    dados = pd.concat([anteriores, conhecidos], ignore_index=True)
    dados = dados.sort_values(["temporada", "data"], kind="stable").reset_index(drop=True)
    temporadas = list(range(temporada - JANELA + 1, temporada + 1))

    # Bloco de cada jogo: posição do jogo na temporada (em ordem cronológica),
    # dividida pelo tamanho de cada bloco. 380 jogos por temporada com 20 times.
    posicao = dados.groupby("temporada").cumcount()
    jogos_por_temporada = 380
    dados["bloco"] = np.minimum(blocos - 1, posicao * blocos // jogos_por_temporada)

    periodos = [(s, b) for s in temporadas for b in range(blocos)
                if s < temporada or b <= dados.loc[dados["temporada"] == temporada, "bloco"].max()]
    pi = {p: i for i, p in enumerate(periodos)}
    times = sorted(set(dados["mandante"]) | set(dados["visitante"]) | set(times_temporada))
    ti = {t: i for i, t in enumerate(times)}
    promovido = np.array([[int((s, t) in promovidos) for t in times] for s, _ in periodos])
    idx_periodo = np.array([pi[(s, b)] for s, b in zip(dados["temporada"], dados["bloco"])])

    # Expectativa do mercado antes de cada jogo (escala 0–1), em escala logit.
    # Jogos sem odds (ex.: resultados digitados à mão) ficam de fora desta parte.
    if "prob_mercado_mandante" in dados.columns:
        esp = (dados["prob_mercado_mandante"] + 0.5 * dados["prob_mercado_empate"]).to_numpy(dtype=float)
    else:
        esp = np.full(len(dados), np.nan)
    tem_odds = ~np.isnan(esp)
    esp = np.clip(esp, 0.02, 0.98)
    return {
        "times": times,
        "temporadas": temporadas,
        "periodos": periodos,
        "muda_temporada": np.array([k > 0 and periodos[k][0] != periodos[k - 1][0] for k in range(len(periodos))]),
        "promovido": promovido,
        "idx_periodo": idx_periodo,
        "idx_mandante": dados["mandante"].map(ti).to_numpy(),
        "idx_visitante": dados["visitante"].map(ti).to_numpy(),
        "gols_mandante": dados["gols_mandante"].to_numpy(dtype=int),
        "gols_visitante": dados["gols_visitante"].to_numpy(dtype=int),
        "mercado_linhas": np.flatnonzero(tem_odds),
        "mercado_logit": np.log(esp[tem_odds] / (1 - esp[tem_odds])),
    }


# =============================================================================
# 2. O MODELO
# =============================================================================

def construir_modelo(d: dict, usar_mercado: bool = False):
    """Define o modelo no PyMC. Cada linha "pm.Distribuição" é uma priori
    (o que achamos antes dos dados) ou, com observed=, a verossimilhança.

    usar_mercado=True acrescenta uma segunda fonte de evidência: as odds dos
    jogos passados. A MESMA força de cada time (ataque − defesa) que explica os
    gols também precisa explicar o que o mercado esperava antes de cada jogo:

        logit(expectativa do mercado) ~ Normal(h + kappa x (força_mandante − força_visitante), sigma)

    O modelo aprende sozinho quanto confiar no mercado (sigma pequeno = confia muito).
    """
    import pymc as pm

    n_per, n_times = len(d["periodos"]), len(d["times"])
    with pm.Model() as modelo:
        # Nível médio de gols e vantagem de casa (em escala log).
        # casa = 0,3 significa ~35% mais gols para o mandante (exp(0,3) = 1,35).
        base = pm.Normal("base", 0, 1)
        casa = pm.Normal("casa", 0.2, 0.2)

        # Quão diferentes os times são entre si, e quanto da força passa de um
        # ano para o outro. Priori Beta(8, 2): esperamos persistência alta (~0,8),
        # mas deixamos os dados decidirem.
        dispersao = pm.HalfNormal("dispersao_forcas", 0.5)
        persistencia = pm.Beta("persistencia", 8, 2)
        # Com essa fórmula, a dispersão das forças fica estável ao longo dos anos
        mudanca = pm.Deterministic("mudanca_anual", dispersao * pm.math.sqrt(1 - persistencia ** 2))
        # Dentro da temporada (só com blocos > 1): passeio aleatório entre blocos
        tem_blocos = any(not x for x in d["muda_temporada"][1:])
        intra = pm.HalfNormal("mudanca_intra", 0.1) if tem_blocos else None

        # Ajuste para quem acabou de subir
        prom_ataque = pm.Normal("promovido_ataque", 0, 0.5)
        prom_defesa = pm.Normal("promovido_defesa", 0, 0.5)

        def forcas_ao_longo_do_tempo(nome):
            # Parametrização "não centrada": sorteamos z ~ Normal(0, 1) e
            # construímos as forças a partir dele. É matematicamente igual,
            # mas facilita muito a vida do amostrador NUTS.
            z = pm.Normal(f"{nome}_z", 0, 1, shape=(n_per, n_times))
            periodos = [dispersao * z[0]]
            for k in range(1, n_per):
                if d["muda_temporada"][k]:          # virada de ano: regressão à média
                    periodos.append(persistencia * periodos[-1] + mudanca * z[k])
                else:                               # mesmo ano: pequena mudança
                    periodos.append(periodos[-1] + intra * z[k])
            return pm.Deterministic(nome, pm.math.stack(periodos))

        ataque = forcas_ao_longo_do_tempo("ataque") + prom_ataque * d["promovido"]
        defesa = forcas_ao_longo_do_tempo("defesa") + prom_defesa * d["promovido"]

        s, m, v = d["idx_periodo"], d["idx_mandante"], d["idx_visitante"]
        lambda_m = pm.math.exp(base + casa + ataque[s, m] + defesa[s, v])
        lambda_v = pm.math.exp(base + ataque[s, v] + defesa[s, m])
        pm.Poisson("gols_m", lambda_m, observed=d["gols_mandante"])
        pm.Poisson("gols_v", lambda_v, observed=d["gols_visitante"])

        if usar_mercado and len(d["mercado_linhas"]):
            forca = ataque - defesa
            r = d["mercado_linhas"]
            h = pm.Normal("mercado_casa", 0.3, 0.3)
            # kappa precisa ser POSITIVO (time mais forte => mercado espera mais dele).
            # Com Normal(1, 1), algumas cadeias ficavam presas num "espelho" ruim
            # (kappa ≈ −5, forças quase zeradas) e o r_hat ia a 1,5. LogNormal(0, 0,5)
            # proíbe o sinal errado e ainda deixa kappa entre ~0,4 e ~2,7.
            kappa = pm.LogNormal("mercado_kappa", 0, 0.5)
            sigma = pm.HalfNormal("mercado_sigma", 0.5)
            pm.Normal("mercado", h + kappa * (forca[s[r], m[r]] - forca[s[r], v[r]]), sigma,
                      observed=d["mercado_logit"])
    return modelo


def ajustar(modelo, semente: int = 1, amostras: int = AMOSTRAS, aquecimento: int = AQUECIMENTO):
    """Roda o MCMC. Usa o nutpie se estiver instalado (mais rápido e sem
    compilador C); senão, o NUTS padrão do PyMC."""
    import pymc as pm

    extra = {"nuts_sampler": "nutpie"} if importlib.util.find_spec("nutpie") else {}
    with modelo:
        return pm.sample(amostras, tune=aquecimento, chains=CADEIAS, random_seed=semente,
                         progressbar=False, **extra)


def diagnosticar(posteriori) -> dict:
    import arviz as az

    nomes = [n for n in ["base", "casa", "dispersao_forcas", "persistencia", "mudanca_intra",
                         "promovido_ataque", "promovido_defesa", "mercado_casa", "mercado_kappa",
                         "mercado_sigma", "ataque_z", "defesa_z"] if n in posteriori.posterior]
    resumo = az.summary(posteriori, var_names=nomes)
    return {"r_hat_max": float(resumo["r_hat"].max()),
            "ess_min": float(resumo["ess_bulk"].min()),
            "divergencias": int(posteriori.sample_stats["diverging"].sum())}


def sorteios_da_temporada(posteriori, d: dict) -> dict:
    """Extrai, para o ÚLTIMO período da janela (o bloco mais recente da
    temporada atual), os sorteios de base, casa, ataque e defesa de cada time
    (já com o efeito promovido)."""
    post = posteriori.posterior
    k = len(d["periodos"]) - 1

    def achatar(nome):        # (cadeias, sorteios, ...) -> (cadeias * sorteios, ...)
        x = post[nome].values
        return x.reshape(-1, *x.shape[2:])

    return {
        "base": achatar("base"),
        "casa": achatar("casa"),
        "ataque": achatar("ataque")[:, k, :] + achatar("promovido_ataque")[:, None] * d["promovido"][k],
        "defesa": achatar("defesa")[:, k, :] + achatar("promovido_defesa")[:, None] * d["promovido"][k],
        "times": d["times"],
    }


# =============================================================================
# 3. SIMULAÇÃO COM A POSTERIORI
# =============================================================================

def simular_restante(sorteios: dict, tabela: pd.DataFrame, jogos: pd.DataFrame,
                     n: int, rng: np.random.Generator) -> dict:
    """Cada simulação escolhe UM sorteio da posteriori (uma versão plausível
    das forças) e joga todos os jogos restantes com ela, sorteando os gols
    de uma Poisson. A incerteza sobre as forças entra naturalmente aqui."""
    times = sorted(tabela.index)
    col = [sorteios["times"].index(t) for t in times]          # forças na ordem da tabela
    escolha = rng.integers(0, len(sorteios["base"]), size=n)   # um sorteio por simulação
    base, casa = sorteios["base"][escolha], sorteios["casa"][escolha]
    ataque, defesa = sorteios["ataque"][escolha][:, col], sorteios["defesa"][escolha][:, col]
    idx = {t: i for i, t in enumerate(times)}

    def matriz(coluna):
        return np.tile(tabela.loc[times, coluna].to_numpy(dtype=float), (n, 1))
    pontos, vitorias = matriz("pontos"), matriz("vitorias")
    gols_pro, gols_contra = matriz("gols_pro"), matriz("gols_contra")

    for _, jogo in jogos.iterrows():
        m, v = idx[jogo["mandante"]], idx[jogo["visitante"]]
        if jogo["conhecido"]:
            gm = np.full(n, int(jogo["gols_mandante"]))
            gv = np.full(n, int(jogo["gols_visitante"]))
        else:
            gm = rng.poisson(np.exp(base + casa + ataque[:, m] + defesa[:, v]))
            gv = rng.poisson(np.exp(base + ataque[:, v] + defesa[:, m]))
        venceu_m, empate, venceu_v = gm > gv, gm == gv, gm < gv
        pontos[:, m] += 3 * venceu_m + empate
        pontos[:, v] += 3 * venceu_v + empate
        vitorias[:, m] += venceu_m
        vitorias[:, v] += venceu_v
        gols_pro[:, m] += gm
        gols_contra[:, m] += gv
        gols_pro[:, v] += gv
        gols_contra[:, v] += gm

    posicao = simulacao.posicoes_finais(pontos, vitorias, gols_pro, gols_contra, rng)
    return {"times": times, "posicao": posicao, "pontos": pontos}


# =============================================================================
# 4. 2026
# =============================================================================

def sufixo_versao(mercado: bool, blocos: int) -> str:
    """Nome curto da versão do modelo, usado nos arquivos de saída."""
    return "_bayes" + ("_mercado" if mercado else "") + (f"_blocos{blocos}" if blocos > 1 else "")


def rodar_2026(n: int, mercado: bool = False, blocos: int = 1) -> None:
    partidas = pd.read_csv(PROCESSADOS / "partidas.csv", parse_dates=["data"])
    partidas = partidas.sort_values(["temporada", "data", "id_partida"]).reset_index(drop=True)
    promovidos = conjunto_promovidos(partidas)

    print("=== 1. Dados ===")
    jogos = simulacao.montar_jogos_a_simular()
    # Os resultados manuais (já disputados) também entram no AJUSTE do modelo
    manuais = jogos[jogos["conhecido"]].assign(temporada=TEMPORADA)[
        ["temporada", "data", "mandante", "visitante", "gols_mandante", "gols_visitante"]]
    atuais = partidas[partidas["temporada"] == TEMPORADA]
    conhecidos = pd.concat([atuais, manuais], ignore_index=True)
    anteriores = partidas[partidas["temporada"].between(TEMPORADA - JANELA + 1, TEMPORADA - 1)]
    times_temporada = sorted(set(atuais["mandante"]))
    d = preparar(anteriores, conhecidos, TEMPORADA, times_temporada, promovidos, blocos)
    print(f"  {len(d['gols_mandante'])} jogos de {d['temporadas'][0]}–{TEMPORADA}, {len(d['times'])} times, "
          f"{len(d['periodos'])} períodos; odds usadas: {len(d['mercado_linhas']) if mercado else 0} jogos")

    print("\n=== 2. Ajustando o modelo (MCMC) ===")
    inicio = time.time()
    posteriori = ajustar(construir_modelo(d, mercado))
    diag = diagnosticar(posteriori)
    print(f"  pronto em {time.time() - inicio:.0f} s")
    print(f"  r_hat máximo {diag['r_hat_max']:.3f} (ideal ≤ 1,01)   ess mínimo {diag['ess_min']:.0f} "
          f"(ideal > 400)   divergências {diag['divergencias']} (ideal 0)")
    if diag["r_hat_max"] > 1.01 or diag["divergencias"] > 0:
        print("  AVISO: o amostrador teve problemas; os resultados podem não ser confiáveis.")

    import arviz as az
    print("\n=== 3. O que o modelo aprendeu ===")
    # (O nome das colunas do intervalo muda entre versões do ArviZ: hdi_... ou eti...)
    nomes = [x for x in ["casa", "persistencia", "mudanca_anual", "mudanca_intra", "dispersao_forcas",
                         "promovido_ataque", "promovido_defesa", "mercado_kappa", "mercado_sigma"]
             if x in posteriori.posterior]
    resumo = az.summary(posteriori, var_names=nomes)
    print(resumo[["mean", "sd"] + [c for c in resumo.columns if c.startswith(("hdi", "eti"))]].to_string())

    sorteios = sorteios_da_temporada(posteriori, d)
    col = [d["times"].index(t) for t in times_temporada]
    forca = sorteios["ataque"][:, col] - sorteios["defesa"][:, col]
    forcas = pd.DataFrame({
        "ataque": sorteios["ataque"][:, col].mean(axis=0),
        "defesa": sorteios["defesa"][:, col].mean(axis=0),
        "forca": forca.mean(axis=0),
        "forca_incerteza": forca.std(axis=0),
    }, index=pd.Index(times_temporada, name="time")).sort_values("forca")
    print("\n  Força = ataque − defesa (maior = melhor); incerteza = desvio padrão da posteriori")
    print(forcas.head(8).round(3).to_string())
    sufixo = sufixo_versao(mercado, blocos)
    forcas.round(4).to_csv(PROCESSADOS / f"forcas{sufixo}_{TEMPORADA}.csv")

    print(f"\n=== 4. Simulando o restante ({n:,} vezes) ===".replace(",", "."))
    tabela = simulacao.tabela_atual(atuais)
    rng = np.random.default_rng(simulacao.SEMENTE)
    sim = simular_restante(sorteios, tabela, jogos, n, rng)
    sim.update({"tabela": tabela, "jogos": jogos, "n": n})
    resumo_sim = simulacao.resumir(sim)

    tabela_print = pd.DataFrame({
        "pts hoje": resumo_sim["pontos_hoje"].astype(int),
        "pts finais (média)": resumo_sim["pontos_finais_medio"].round(1),
        "bayesiano": (resumo_sim["prob_rebaixamento"] * 100).round(1),
    })
    etapa1 = PROCESSADOS / f"simulacao_{TEMPORADA}.csv"
    if etapa1.exists():
        tabela_print["etapa 1 (logística)"] = (pd.read_csv(etapa1, index_col="time")["prob_rebaixamento"] * 100).round(1)
    print("\n  Probabilidade de rebaixamento (%)")
    print(tabela_print.head(10).to_string())
    corte = simulacao.pontos_de_corte(sim)
    print(f"\n  Pontos do 17º colocado: mediana {corte.median():.0f} "
          f"(80% entre {corte.quantile(.1):.0f} e {corte.quantile(.9):.0f})")

    resumo_sim.to_csv(PROCESSADOS / f"simulacao_{TEMPORADA}{sufixo}.csv")
    # Diagnóstico salvo à parte: o atualizar.py usa para avisar se o ajuste falhou
    (PROCESSADOS / f"diagnostico{sufixo}_{TEMPORADA}.json").write_text(json.dumps(
        {k: round(float(v), 4) for k, v in diag.items()}, indent=2), encoding="utf-8")
    print(f"\n  arquivos salvos em {PROCESSADOS}")


# =============================================================================
# 5. BACKTEST
# =============================================================================

def rodar_backtest(n: int, mercado: bool = False, blocos: int = 1, amostras: int = AMOSTRAS) -> None:
    """Mesmos cenários do 07_backtest.py (temporadas 2015–2025, cortes em várias
    rodadas), para comparar com a etapa 1 nas mesmas condições."""
    backtest = import_module("07_backtest")
    partidas, _, finais, _ = backtest.carregar()
    promovidos = conjunto_promovidos(partidas)
    caiu = finais.set_index(["temporada", "time"])["rebaixado"]

    print(f"=== Backtest bayesiano: {len(backtest.TEMPORADAS)} temporadas x {len(CORTES_BACKTEST)} cortes ===")
    inicio = time.time()
    linhas, problemas = [], []
    for temporada in backtest.TEMPORADAS:
        da_temporada = partidas[partidas["temporada"] == temporada]
        times = sorted(set(da_temporada["mandante"]) | set(da_temporada["visitante"]))
        anteriores = partidas[partidas["temporada"].between(temporada - JANELA + 1, temporada - 1)]
        for corte in CORTES_BACKTEST:
            n_conhecidos = corte * len(times) // 2
            conhecidos = da_temporada.iloc[:n_conhecidos]
            restantes = da_temporada.iloc[n_conhecidos:][["mandante", "visitante"]].assign(
                conhecido=False, gols_mandante=np.nan, gols_visitante=np.nan).reset_index(drop=True)

            d = preparar(anteriores, conhecidos, temporada, times, promovidos, blocos)
            posteriori = ajustar(construir_modelo(d, mercado), semente=temporada * 100 + corte,
                                 amostras=amostras, aquecimento=amostras)
            diag = diagnosticar(posteriori)
            if diag["r_hat_max"] > 1.01 or diag["divergencias"] > 0:
                problemas.append((temporada, corte, diag))

            tabela = backtest.classificacao_no_corte(conhecidos, times)
            rng = np.random.default_rng(temporada * 100 + corte)
            sim = simular_restante(sorteios_da_temporada(posteriori, d), tabela, restantes, n, rng)
            prob = (sim["posicao"] > len(times) - N_REBAIXADOS).mean(axis=0)
            for t, p in zip(sim["times"], prob):
                linhas.append({"temporada": temporada, "corte": corte, "time": t,
                               "posicao_no_corte": tabela.loc[t, "posicao"],
                               "bayesiano": p, "caiu": caiu[(temporada, t)]})
        print(f"  {temporada} pronto ({time.time() - inicio:.0f} s)")

    prev = pd.DataFrame(linhas)
    prev.to_csv(PROCESSADOS / f"backtest_previsoes{sufixo_versao(mercado, blocos)}.csv", index=False)
    if problemas:
        print(f"\n  AVISO: {len(problemas)} ajustes com diagnóstico ruim: {problemas}")

    # Comparação com a etapa 1 nos MESMOS cenários
    etapa1 = pd.read_csv(PROCESSADOS / "backtest_previsoes.csv")
    juntos = prev.merge(etapa1[["temporada", "corte", "time", "modelo_dinamico", "historico_posicao"]],
                        on=["temporada", "corte", "time"])
    print("\n=== Brier por corte (menor = melhor) ===")
    for periodo, filtro in [("2015–2025", juntos["temporada"] > 0),
                            ("2022–2025 (limpo)", juntos["temporada"] >= 2022)]:
        print(f"\n  [{periodo}]")
        sel = juntos[filtro]
        tab = sel.groupby("corte").apply(lambda g: pd.Series({
            c: ((g[c] - g["caiu"]) ** 2).mean() for c in ["bayesiano", "modelo_dinamico", "historico_posicao"]}),
            include_groups=False)
        tab.loc["todos"] = [((sel[c] - sel["caiu"]) ** 2).mean() for c in tab.columns]
        print(tab.rename(columns={"modelo_dinamico": "etapa 1"}).round(4).to_string())

    print("\n=== Calibração do bayesiano ===")
    print(backtest.calibracao(prev.rename(columns={"bayesiano": "p"}), "p").round(3).to_string())
    print(f"\n  ({time.time() - inicio:.0f} s no total)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo hierárquico bayesiano de gols.")
    parser.add_argument("--n", type=int, default=20000, help="simulações (padrão 20.000)")
    parser.add_argument("--backtest", action="store_true", help="roda o backtest 2015–2025")
    parser.add_argument("--sem-mercado", dest="mercado", action="store_false",
                        help="NÃO usa as odds dos jogos passados (versão original, só gols)")
    parser.add_argument("--blocos", type=int, default=4,
                        help="divide cada temporada em N blocos em que a força pode mudar "
                             "(padrão 4; 1 = força fixa na temporada)")
    parser.add_argument("--amostras", type=int, default=AMOSTRAS,
                        help="sorteios por cadeia no backtest (padrão 1.000; 500 deixa 2x mais rápido)")
    args = parser.parse_args()
    if args.backtest:
        rodar_backtest(min(args.n, 5000), args.mercado, args.blocos, args.amostras)
    else:
        rodar_2026(args.n, args.mercado, args.blocos)


if __name__ == "__main__":
    main()
