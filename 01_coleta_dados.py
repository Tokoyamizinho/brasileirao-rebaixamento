"""
01_coleta_dados.py
==================

Etapa 1 do projeto "Previsão dos rebaixados do Brasileirão 2026".

O QUE ESTE SCRIPT FAZ
---------------------
1. Baixa os resultados e as odds de todas as partidas da Série A desde 2012
   (fonte principal: football-data.co.uk, arquivo BRA.csv).
2. Padroniza nomes de colunas e de times (cada fonte escreve os nomes de um jeito).
3. Converte as odds em probabilidades implícitas, removendo a margem da casa.
4. Numera os jogos de cada time na temporada (1º jogo, 2º jogo, ...). Isso nos
   permite depois "congelar" uma temporada antiga na rodada 30, por exemplo.
5. Calcula a classificação final de cada temporada encerrada e marca quem caiu.
6. Monta a lista de jogos que ainda faltam em 2026.
7. Roda checagens de qualidade e imprime um resumo.

COMO RODAR
----------
No terminal, dentro da pasta do projeto:

    python 01_coleta_dados.py

Opções:
    --forcar-download        baixa de novo mesmo que o arquivo já exista em data/raw
    --arquivo-local CAMINHO  usa um BRA.csv que você já tem, sem baixar nada
    --conferir               compara os placares com uma segunda fonte (GitHub,
                             2003–2024) para achar erros nos dados

Requisitos: Python 3.9+ e pandas (pip install pandas).

ARQUIVOS GERADOS (pasta data/processed)
--------------------------------------
partidas.csv          uma linha por partida já disputada (2012 até hoje)
tabelas_finais.csv    classificação final de cada temporada encerrada,
                      com a coluna `rebaixado` (1 = caiu, 0 = ficou)
jogos_restantes_2026.csv   partidas de 2026 que ainda não aconteceram

Ao trabalhar em Jupyter, você pode importar as funções deste arquivo:
    from importlib import import_module
    coleta = import_module("01_coleta_dados")
    partidas = coleta.carregar_football_data("data/raw/BRA.csv")
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================
# Tudo que pode mudar fica aqui em cima, para não ter que caçar no meio do código.

TEMPORADA_ATUAL = 2026
N_REBAIXADOS = 4          # os 4 últimos caem (posições 17 a 20)
N_TIMES = 20

# Fonte principal: resultados + odds de fechamento, Série A desde 2012.
URL_FOOTBALL_DATA = "https://www.football-data.co.uk/new/BRA.csv"

# Fonte secundária (só para conferência): dataset público no GitHub, 2003–2024.
URL_GITHUB = (
    "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/"
    "master/campeonato-brasileiro-full.csv"
)

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_RAW = PASTA_PROJETO / "data" / "raw"
PASTA_PROCESSADOS = PASTA_PROJETO / "data" / "processed"

# Punições do tribunal (STJD) que mudaram a tabela final e que NÃO aparecem
# quando somamos os pontos jogo a jogo. Em 2013, Flamengo e Portuguesa perderam
# 4 pontos por escalar jogadores irregulares; por isso a Portuguesa caiu no lugar
# do Fluminense. Se você descobrir outro caso, é só acrescentar aqui.
PUNICOES = {
    2013: {"Portuguesa": -4, "Flamengo": -4},
}

# Rebaixados oficiais, usados SÓ para conferir se nosso cálculo da tabela está
# certo. Se a conferência apontar diferença, ou o dado está errado ou falta
# alguma punição em PUNICOES.
REBAIXADOS_OFICIAIS = {
    2012: {"Atlético-GO", "Figueirense", "Palmeiras", "Sport"},
    2013: {"Portuguesa", "Vasco", "Ponte Preta", "Náutico"},
    2014: {"Vitória", "Bahia", "Botafogo", "Criciúma"},
    2015: {"Avaí", "Vasco", "Goiás", "Joinville"},
    2016: {"Internacional", "Figueirense", "Santa Cruz", "América-MG"},
    2017: {"Coritiba", "Avaí", "Ponte Preta", "Atlético-GO"},
    2018: {"Sport", "América-MG", "Vitória", "Paraná"},
    2019: {"Cruzeiro", "CSA", "Chapecoense", "Avaí"},
    2020: {"Vasco", "Goiás", "Coritiba", "Botafogo"},
    2021: {"Grêmio", "Bahia", "Sport", "Chapecoense"},
    2022: {"Ceará", "Atlético-GO", "Avaí", "Juventude"},
    2023: {"Santos", "Goiás", "Coritiba", "América-MG"},
    2024: {"Athletico-PR", "Criciúma", "Atlético-GO", "Cuiabá"},
    2025: {"Sport", "Juventude", "Fortaleza", "Ceará"},
}

# -----------------------------------------------------------------------------
# Nomes dos times
# -----------------------------------------------------------------------------
# Cada fonte escreve o nome de um jeito ("Flamengo RJ", "Atletico-MG",
# "Athletico-PR", "Sport Recife"...). Definimos um nome "oficial" para cada clube
# e um dicionário de apelidos que apontam para ele.

NOMES_OFICIAIS = [
    "América-MG", "América-RN", "Athletico-PR", "Atlético-GO", "Atlético-MG",
    "Avaí", "Bahia", "Barueri", "Botafogo", "Bragantino", "Brasiliense", "CSA",
    "Ceará", "Chapecoense", "Corinthians", "Coritiba", "Criciúma", "Cruzeiro",
    "Cuiabá", "Figueirense", "Flamengo", "Fluminense", "Fortaleza", "Goiás",
    "Grêmio", "Grêmio Prudente", "Guarani", "Internacional", "Ipatinga",
    "Joinville", "Juventude", "Mirassol", "Náutico", "Palmeiras", "Paraná",
    "Paysandu", "Ponte Preta", "Portuguesa", "Remo", "Santa Cruz", "Santo André",
    "Santos", "São Caetano", "São Paulo", "Sport", "Vasco", "Vitória",
]

# Apelidos extras (já "normalizados": minúsculas, sem acento, sem pontuação).
# Se o script avisar "time não reconhecido", acrescente o apelido aqui.
APELIDOS = {
    "flamengo rj": "Flamengo",
    "botafogo rj": "Botafogo",
    "atletico mineiro": "Atlético-MG",
    "atletico pr": "Athletico-PR",
    "athletico paranaense": "Athletico-PR",
    "atletico paranaense": "Athletico-PR",
    "atletico goianiense": "Atlético-GO",
    "america mineiro": "América-MG",
    "sport recife": "Sport",
    "chapecoense sc": "Chapecoense",
    "rb bragantino": "Bragantino",
    "red bull bragantino": "Bragantino",
    "vasco da gama": "Vasco",
    "remo pa": "Remo",
    "clube do remo": "Remo",
    "mirassol sp": "Mirassol",
    "ceara sc": "Ceará",
}


def normalizar_texto(texto: str) -> str:
    """Deixa o texto em um formato comparável: sem acento, minúsculo e com
    qualquer pontuação trocada por espaço.

    Exemplo: "Atlético-MG" -> "atletico mg"
    """
    sem_acento = (
        unicodedata.normalize("NFKD", str(texto))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", " ", sem_acento.lower()).strip()


# Monta o dicionário final: cada nome oficial normalizado aponta para si mesmo,
# e depois acrescentamos os apelidos.
MAPA_NOMES = {normalizar_texto(n): n for n in NOMES_OFICIAIS}
MAPA_NOMES.update(APELIDOS)


def padronizar_nome_time(nome: str, nao_reconhecidos: set | None = None) -> str:
    """Converte o nome de um time para o nome oficial do projeto.

    Se o nome não estiver no dicionário, devolve o original e o anota em
    `nao_reconhecidos`, para avisarmos no final.
    """
    chave = normalizar_texto(nome)
    if chave in MAPA_NOMES:
        return MAPA_NOMES[chave]
    if nao_reconhecidos is not None:
        nao_reconhecidos.add(nome)
    return nome


# =============================================================================
# 2. DOWNLOAD
# =============================================================================

def baixar_arquivo(url: str, destino: Path, forcar: bool = False) -> Path:
    """Baixa `url` para `destino`. Se o arquivo já existir, não baixa de novo
    (a menos que `forcar=True`). Isso evita sobrecarregar o site e deixa o
    script rápido quando rodamos várias vezes."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not forcar:
        print(f"  usando cópia local: {destino.name}")
        return destino

    print(f"  baixando {url} ...")
    # Alguns sites recusam requisições sem "User-Agent"; mandamos um simples.
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        destino.write_bytes(resposta.read())
    print(f"  salvo em {destino}")
    return destino


# =============================================================================
# 3. LEITURA E PADRONIZAÇÃO DA FONTE PRINCIPAL
# =============================================================================

# O football-data já mudou o nome das colunas de odds ao longo dos anos.
# Listamos os trios possíveis em ordem de preferência:
#   PSC* = Pinnacle no fechamento (a casa mais "afiada", melhor referência)
#   AvgC* = média do mercado no fechamento
#   PS* / P* = Pinnacle (abertura)    Avg* = média do mercado (abertura)
CANDIDATOS_ODDS = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_fechamento"),
    ("AvgCH", "AvgCD", "AvgCA", "media_fechamento"),
    ("PSH", "PSD", "PSA", "pinnacle"),
    ("PH", "PD", "PA", "pinnacle"),
    ("AvgH", "AvgD", "AvgA", "media"),
]


def escolher_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as colunas odd_mandante, odd_empate e odd_visitante.

    Para cada partida, pega o primeiro trio de odds disponível, seguindo a
    ordem de CANDIDATOS_ODDS. Também guarda em `odds_fonte` de onde veio.
    """
    df = df.copy()
    df["odd_mandante"] = float("nan")
    df["odd_empate"] = float("nan")
    df["odd_visitante"] = float("nan")
    df["odds_fonte"] = pd.NA

    for col_m, col_e, col_v, nome_fonte in CANDIDATOS_ODDS:
        if not {col_m, col_e, col_v}.issubset(df.columns):
            continue  # esse trio não existe no arquivo
        # Linhas que ainda não têm odd e que têm este trio completo
        faltando = df["odd_mandante"].isna()
        tem_trio = df[[col_m, col_e, col_v]].notna().all(axis=1)
        usar = faltando & tem_trio
        df.loc[usar, "odd_mandante"] = df.loc[usar, col_m]
        df.loc[usar, "odd_empate"] = df.loc[usar, col_e]
        df.loc[usar, "odd_visitante"] = df.loc[usar, col_v]
        df.loc[usar, "odds_fonte"] = nome_fonte
    return df


def odds_para_probabilidades(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma odds em probabilidades.

    A probabilidade "crua" de uma odd é 1/odd. Mas, somando as três (mandante,
    empate, visitante), o total passa de 100%: essa sobra é a margem da casa
    (overround). Para ter probabilidades que somam 1, dividimos cada uma pelo
    total. É o método mais simples (proporcional) e serve bem para começar.

    Exemplo: odds 2.00 / 3.40 / 3.80
        crus: 0.500 + 0.294 + 0.263 = 1.057  (margem de 5,7%)
        justas: 0.473, 0.278, 0.249
    """
    df = df.copy()
    inv_m = 1 / df["odd_mandante"]
    inv_e = 1 / df["odd_empate"]
    inv_v = 1 / df["odd_visitante"]
    soma = inv_m + inv_e + inv_v

    df["margem_casa"] = soma - 1
    df["prob_mercado_mandante"] = inv_m / soma
    df["prob_mercado_empate"] = inv_e / soma
    df["prob_mercado_visitante"] = inv_v / soma
    return df


def carregar_football_data(caminho: Path, nao_reconhecidos: set | None = None) -> pd.DataFrame:
    """Lê o BRA.csv do football-data e devolve uma tabela limpa, com
    colunas em português e nomes de times padronizados."""
    bruto = pd.read_csv(caminho)
    bruto.columns = bruto.columns.str.strip()  # tira espaços escondidos nos nomes

    # O arquivo tem só a Série A, mas filtramos por garantia caso a coluna exista.
    if "League" in bruto.columns:
        bruto = bruto[bruto["League"].astype(str).str.contains("Serie A", case=False)]

    bruto = escolher_odds(bruto)

    df = pd.DataFrame({
        "temporada": pd.to_numeric(bruto["Season"], errors="coerce"),
        # dayfirst=True porque a data vem como dia/mês/ano
        "data": pd.to_datetime(bruto["Date"], dayfirst=True, errors="coerce"),
        "mandante": bruto["Home"],
        "visitante": bruto["Away"],
        "gols_mandante": pd.to_numeric(bruto["HG"], errors="coerce"),
        "gols_visitante": pd.to_numeric(bruto["AG"], errors="coerce"),
        "odd_mandante": bruto["odd_mandante"],
        "odd_empate": bruto["odd_empate"],
        "odd_visitante": bruto["odd_visitante"],
        "odds_fonte": bruto["odds_fonte"],
    })

    # Jogos sem placar (adiados, ou linhas vazias no fim do arquivo) saem daqui.
    # Os jogos que faltam em 2026 serão reconstruídos mais adiante.
    df = df.dropna(subset=["temporada", "data", "gols_mandante", "gols_visitante"])
    df["temporada"] = df["temporada"].astype(int)
    df["gols_mandante"] = df["gols_mandante"].astype(int)
    df["gols_visitante"] = df["gols_visitante"].astype(int)

    # Nomes padronizados
    df["mandante"] = df["mandante"].apply(padronizar_nome_time, nao_reconhecidos=nao_reconhecidos)
    df["visitante"] = df["visitante"].apply(padronizar_nome_time, nao_reconhecidos=nao_reconhecidos)

    # Resultado do ponto de vista do mandante: M = mandante venceu, E = empate,
    # V = visitante venceu. Esse será o ALVO do nosso modelo de partidas.
    df["resultado"] = "E"
    df.loc[df["gols_mandante"] > df["gols_visitante"], "resultado"] = "M"
    df.loc[df["gols_mandante"] < df["gols_visitante"], "resultado"] = "V"

    df = odds_para_probabilidades(df)
    df = df.sort_values(["temporada", "data", "mandante"]).reset_index(drop=True)
    return df


# =============================================================================
# 4. NÚMERO DO JOGO DE CADA TIME NA TEMPORADA
# =============================================================================

def adicionar_numero_do_jogo(partidas: pd.DataFrame) -> pd.DataFrame:
    """Para cada partida, diz qual era o jogo de número N do mandante e do
    visitante naquela temporada (1, 2, ..., 38).

    Por que isso importa: o football-data não traz a rodada oficial, e jogos
    adiados bagunçam a ordem. Com o número do jogo de cada time, conseguimos
    reproduzir a situação "depois de 30 jogos" de qualquer temporada passada.
    """
    df = partidas.copy()
    df["id_partida"] = range(len(df))

    # "Formato longo": cada partida vira duas linhas, uma por time.
    lado_mandante = df[["id_partida", "temporada", "data", "mandante"]].rename(columns={"mandante": "time"})
    lado_mandante["lado"] = "mandante"
    lado_visitante = df[["id_partida", "temporada", "data", "visitante"]].rename(columns={"visitante": "time"})
    lado_visitante["lado"] = "visitante"
    longo = pd.concat([lado_mandante, lado_visitante]).sort_values(["temporada", "time", "data", "id_partida"])

    # cumcount() numera as linhas dentro de cada grupo: 0, 1, 2... somamos 1.
    longo["jogo_n"] = longo.groupby(["temporada", "time"]).cumcount() + 1

    # Volta para o formato "uma linha por partida"
    largo = longo.pivot(index="id_partida", columns="lado", values="jogo_n")
    df["jogo_n_mandante"] = df["id_partida"].map(largo["mandante"]).astype(int)
    df["jogo_n_visitante"] = df["id_partida"].map(largo["visitante"]).astype(int)
    return df


# =============================================================================
# 5. CLASSIFICAÇÃO
# =============================================================================

def calcular_classificacao(partidas: pd.DataFrame, punicoes: dict | None = None) -> pd.DataFrame:
    """Calcula a tabela de UMA temporada a partir das partidas.

    Critérios de desempate do Brasileirão: pontos, vitórias, saldo de gols,
    gols marcados. (Há outros depois desses, como confronto direto, mas quase
    nunca chegam a ser usados.)
    """
    mand = pd.DataFrame({
        "time": partidas["mandante"],
        "gp": partidas["gols_mandante"],
        "gc": partidas["gols_visitante"],
    })
    visi = pd.DataFrame({
        "time": partidas["visitante"],
        "gp": partidas["gols_visitante"],
        "gc": partidas["gols_mandante"],
    })
    jogos = pd.concat([mand, visi])
    jogos["v"] = (jogos["gp"] > jogos["gc"]).astype(int)
    jogos["e"] = (jogos["gp"] == jogos["gc"]).astype(int)
    jogos["d"] = (jogos["gp"] < jogos["gc"]).astype(int)

    tabela = jogos.groupby("time").agg(
        jogos=("gp", "size"), vitorias=("v", "sum"), empates=("e", "sum"),
        derrotas=("d", "sum"), gols_pro=("gp", "sum"), gols_contra=("gc", "sum"),
    )
    tabela["saldo"] = tabela["gols_pro"] - tabela["gols_contra"]
    tabela["pontos"] = 3 * tabela["vitorias"] + tabela["empates"]

    tabela["punicao"] = 0
    for time, pts in (punicoes or {}).items():
        if time in tabela.index:
            tabela.loc[time, "punicao"] = pts
    tabela["pontos"] += tabela["punicao"]

    tabela = tabela.sort_values(
        ["pontos", "vitorias", "saldo", "gols_pro"], ascending=False
    ).reset_index()
    tabela.insert(0, "posicao", range(1, len(tabela) + 1))
    return tabela


def calcular_tabelas_finais(partidas: pd.DataFrame) -> pd.DataFrame:
    """Classificação final de todas as temporadas JÁ ENCERRADAS, com a coluna
    `rebaixado` (1 = terminou entre os 4 últimos)."""
    tabelas = []
    encerradas = sorted(t for t in partidas["temporada"].unique() if t < TEMPORADA_ATUAL)
    for temporada in encerradas:
        jogos_temp = partidas[partidas["temporada"] == temporada]
        tab = calcular_classificacao(jogos_temp, PUNICOES.get(temporada))
        tab.insert(0, "temporada", temporada)
        n_times = len(tab)
        tab["rebaixado"] = (tab["posicao"] > n_times - N_REBAIXADOS).astype(int)
        tabelas.append(tab)
    return pd.concat(tabelas, ignore_index=True)


# =============================================================================
# 6. JOGOS QUE FALTAM EM 2026
# =============================================================================

def montar_jogos_restantes(partidas: pd.DataFrame, temporada: int = TEMPORADA_ATUAL) -> pd.DataFrame:
    """No Brasileirão, todo time enfrenta todos os outros duas vezes: uma em casa
    e outra fora. Então os jogos que faltam são exatamente os pares
    (mandante, visitante) que ainda não aparecem nos resultados.

    Isso já inclui jogos adiados, que ficam "para trás" na tabela oficial.
    Ainda não temos as datas: para a simulação, só precisamos saber quem
    joga contra quem e onde.
    """
    jogos_temp = partidas[partidas["temporada"] == temporada]
    times = sorted(set(jogos_temp["mandante"]) | set(jogos_temp["visitante"]))
    ja_jogados = set(zip(jogos_temp["mandante"], jogos_temp["visitante"]))

    restantes = [
        {"temporada": temporada, "mandante": m, "visitante": v}
        for m in times for v in times
        if m != v and (m, v) not in ja_jogados
    ]
    return pd.DataFrame(restantes)


# =============================================================================
# 7. CHECAGENS DE QUALIDADE
# =============================================================================
# Regra de ouro: nunca confie em um dado antes de olhar para ele. Cada checagem
# imprime OK ou um AVISO explicando o que pode estar errado.

def checar_qualidade(partidas: pd.DataFrame, tabelas_finais: pd.DataFrame) -> None:
    print("\n=== Checagens de qualidade ===")

    # (a) Cada temporada encerrada deveria ter 20 times e 380 jogos.
    por_temp = partidas.groupby("temporada").agg(
        jogos=("resultado", "size"),
        times=("mandante", "nunique"),
        com_odds=("odd_mandante", lambda s: s.notna().mean()),
    )
    for temporada, linha in por_temp.iterrows():
        status = "OK"
        if temporada < TEMPORADA_ATUAL and (linha["jogos"] != 380 or linha["times"] != N_TIMES):
            status = "AVISO: esperava 20 times e 380 jogos"
            if temporada == 2016 and linha["jogos"] == 379:
                status = "OK (Chapecoense x Atlético-MG não foi disputado após o acidente aéreo)"
        print(f"  {temporada}: {int(linha['jogos']):>3} jogos, {int(linha['times'])} times, "
              f"{linha['com_odds']:.0%} com odds  -> {status}")

    # (b) Jogo duplicado (mesmo mandante x visitante duas vezes na temporada)?
    duplicados = partidas.duplicated(subset=["temporada", "mandante", "visitante"], keep=False)
    if duplicados.any():
        print(f"\n  AVISO: {duplicados.sum()} linhas com confronto repetido na mesma temporada:")
        print(partidas.loc[duplicados, ["temporada", "data", "mandante", "visitante"]].to_string(index=False))
    else:
        print("\n  Sem confrontos duplicados: OK")

    # (c) Odds estranhas (abaixo de 1.01 é impossível na prática)
    odds_ruins = (partidas[["odd_mandante", "odd_empate", "odd_visitante"]] < 1.01).any(axis=1)
    print(f"  Odds inválidas (< 1.01): {odds_ruins.sum()}")
    print(f"  Margem média das casas: {partidas['margem_casa'].mean():.1%}")

    # (d) Nossa tabela calculada bate com os rebaixados oficiais?
    print("\n  Rebaixados calculados x oficiais:")
    for temporada, oficiais in REBAIXADOS_OFICIAIS.items():
        tab = tabelas_finais[tabelas_finais["temporada"] == temporada]
        if tab.empty:
            continue
        calculados = set(tab.loc[tab["rebaixado"] == 1, "time"])
        if calculados == oficiais:
            print(f"    {temporada}: OK")
        else:
            print(f"    {temporada}: DIFERENTE  calculado={sorted(calculados)}  oficial={sorted(oficiais)}")


def conferir_com_fonte_secundaria(partidas: pd.DataFrame, caminho_github: Path) -> None:
    """Compara os placares com o dataset do GitHub (2003–2024).

    Casamos as partidas por temporada + mandante + visitante, e NÃO pela data.
    Motivo: o football-data registra a data no horário do Reino Unido, então um
    jogo às 21h de Brasília aparece lá com a data do dia seguinte. Como cada
    confronto acontece uma única vez por temporada, essa chave é suficiente.
    """
    print("\n=== Conferência com a fonte secundária (GitHub) ===")
    outra = pd.read_csv(caminho_github)
    datas = pd.to_datetime(outra["data"], dayfirst=True, errors="coerce")
    outra = pd.DataFrame({
        # A temporada 2020 terminou em fevereiro de 2021 (pandemia). Por isso,
        # jogos de janeiro a março contam para a temporada do ano anterior.
        "temporada": datas.dt.year - (datas.dt.month <= 3).astype(int),
        "mandante": outra["mandante"].apply(padronizar_nome_time),
        "visitante": outra["visitante"].apply(padronizar_nome_time),
        "gm_github": pd.to_numeric(outra["mandante_Placar"], errors="coerce"),
        "gv_github": pd.to_numeric(outra["visitante_Placar"], errors="coerce"),
    })

    periodo = partidas[partidas["temporada"] <= outra["temporada"].max()]
    juntos = periodo.merge(outra, on=["temporada", "mandante", "visitante"], how="left")
    encontrados = juntos["gm_github"].notna()
    print(f"  Partidas do período comum: {len(periodo)}")
    print(f"  Encontradas na outra fonte: {encontrados.sum()} ({encontrados.mean():.1%})")

    divergentes = juntos[encontrados & (
        (juntos["gols_mandante"] != juntos["gm_github"]) |
        (juntos["gols_visitante"] != juntos["gv_github"])
    )]
    print(f"  Placares divergentes: {len(divergentes)}")
    if len(divergentes):
        print(divergentes[["data", "mandante", "visitante", "gols_mandante",
                           "gols_visitante", "gm_github", "gv_github"]].head(20).to_string(index=False))
    # Partidas não encontradas costumam ser diferença no nome de um time.
    if (~encontrados).any():
        print("\n  Exemplos não encontrados (provável diferença de nome):")
        print(juntos.loc[~encontrados, ["temporada", "mandante", "visitante"]].head(10).to_string(index=False))


# =============================================================================
# 8. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta e limpeza dos dados do Brasileirão.")
    parser.add_argument("--forcar-download", action="store_true",
                        help="baixa os arquivos de novo mesmo se já existirem")
    parser.add_argument("--arquivo-local", type=Path, default=None,
                        help="caminho de um BRA.csv já baixado")
    parser.add_argument("--conferir", action="store_true",
                        help="compara placares com a fonte secundária do GitHub")
    args = parser.parse_args()

    print("=== 1. Download ===")
    if args.arquivo_local:
        caminho_bra = args.arquivo_local
        print(f"  usando arquivo informado: {caminho_bra}")
    else:
        caminho_bra = baixar_arquivo(URL_FOOTBALL_DATA, PASTA_RAW / "BRA.csv", args.forcar_download)

    print("\n=== 2. Leitura e padronização ===")
    nao_reconhecidos: set = set()
    partidas = carregar_football_data(caminho_bra, nao_reconhecidos)
    partidas = adicionar_numero_do_jogo(partidas)
    print(f"  {len(partidas)} partidas, temporadas {partidas['temporada'].min()}–{partidas['temporada'].max()}")
    if nao_reconhecidos:
        print("  AVISO: times não reconhecidos (acrescente em APELIDOS):")
        for nome in sorted(nao_reconhecidos):
            print(f"    - {nome!r}  (forma normalizada: {normalizar_texto(nome)!r})")

    print("\n=== 3. Tabelas finais e rebaixados ===")
    tabelas_finais = calcular_tabelas_finais(partidas)
    print(f"  {tabelas_finais['temporada'].nunique()} temporadas encerradas, "
          f"{tabelas_finais['rebaixado'].sum()} rebaixamentos")

    print(f"\n=== 4. Situação de {TEMPORADA_ATUAL} ===")
    jogos_atuais = partidas[partidas["temporada"] == TEMPORADA_ATUAL]
    if jogos_atuais.empty:
        print(f"  AVISO: nenhum jogo de {TEMPORADA_ATUAL} no arquivo.")
        restantes = pd.DataFrame(columns=["temporada", "mandante", "visitante"])
    else:
        restantes = montar_jogos_restantes(partidas)
        tabela_atual = calcular_classificacao(jogos_atuais)
        print(f"  {len(jogos_atuais)} jogos disputados, {len(restantes)} restantes "
              f"(total {len(jogos_atuais) + len(restantes)}, esperado 380)")
        print(f"  Último jogo no arquivo: {jogos_atuais['data'].max():%d/%m/%Y}")
        print("\n  Parte de baixo da tabela atual:")
        print(tabela_atual.tail(8)[["posicao", "time", "jogos", "pontos", "vitorias", "saldo"]]
              .to_string(index=False))

    checar_qualidade(partidas, tabelas_finais)

    if args.conferir:
        caminho_github = baixar_arquivo(URL_GITHUB, PASTA_RAW / "brasileirao_github_2003_2024.csv",
                                        args.forcar_download)
        conferir_com_fonte_secundaria(partidas, caminho_github)

    print("\n=== 5. Salvando ===")
    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    partidas.to_csv(PASTA_PROCESSADOS / "partidas.csv", index=False)
    tabelas_finais.to_csv(PASTA_PROCESSADOS / "tabelas_finais.csv", index=False)
    restantes.to_csv(PASTA_PROCESSADOS / f"jogos_restantes_{TEMPORADA_ATUAL}.csv", index=False)
    print(f"  arquivos salvos em {PASTA_PROCESSADOS}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as erro:
        sys.exit(f"\nNão consegui baixar os dados ({erro}). Verifique a internet, ou baixe "
                 f"o arquivo manualmente em {URL_FOOTBALL_DATA} e rode com --arquivo-local.")
