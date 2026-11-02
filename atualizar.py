"""
atualizar.py
============

Atualiza a previsão depois de cada rodada, DE PONTA A PONTA, com um único
comando. Pensado para rodar sozinho (Agendador de Tarefas do Windows ou, mais
tarde, GitHub Actions), então ele PARA e avisa em vez de publicar algo errado.

O QUE ELE FAZ, NA ORDEM
-----------------------
1. Baixa o BRA.csv do football-data (01_coleta_dados.py).
2. Confere os dados de 2026 (football-data + resultados manuais):
   20 times, nenhum jogo repetido, nenhum time com mais de 38 jogos,
   jogos conhecidos + restantes = 380, e nenhum placar manual que contradiga
   o football-data. Se algo falhar, para aqui.
3. Se não há nada novo desde a última atualização, encerra (use --forcar
   para rodar mesmo assim).
4. Recalcula as features (02) e simula com o modelo principal (06).
   Na primeira vez (ou se o modelo salvo for de uma versão antiga), também
   otimiza o Elo do mercado (02 --otimizar) e treina a logística (03).
5. Ajusta o modelo bayesiano e simula com ele (05). Se o bayesiano falhar,
   a atualização continua só com o modelo principal, com um aviso.
6. Confere as saídas (as probabilidades de queda somam 4, 20 times...).
7. Guarda uma linha por time em data/processed/historico_previsoes.csv.
8. Gera a página a partir do modelo em pagina/: apresentacao.html (para o Claude)
   e docs/index.html (documento completo, para o GitHub Pages).
9. Com --publicar, faz commit e push para o GitHub.

COMO RODAR
----------
    python atualizar.py                  # atualização completa (~5 min)
    python atualizar.py --forcar         # roda mesmo sem jogos novos
    python atualizar.py --sem-download   # usa o BRA.csv que já está em data/raw
    python atualizar.py --sem-bayes      # pula o bayesiano (economiza ~3 min)
    python atualizar.py --retreinar      # treina de novo a logística (03)
    python atualizar.py --so-pagina      # só gera a página com os arquivos atuais
    python atualizar.py --publicar       # no fim, commit + push para o GitHub (GitHub Pages)
    python atualizar.py --agendar 09:00  # cria a tarefa diária no Agendador do Windows
    python atualizar.py --agendar 09:00 --publicar   # tarefa diária que também publica
    python atualizar.py --desagendar     # remove a tarefa

Cada execução grava um registro completo em logs/atualizacao_AAAA-MM-DD_HHMM.log.
O código de saída é 0 quando deu certo (ou não havia nada novo) e 1 quando
alguma conferência falhou: o Agendador de Tarefas mostra isso em "Último resultado".

AGENDAMENTO (Windows)
---------------------
--agendar cria uma tarefa diária no Agendador de Tarefas que roda este arquivo
com o MESMO Python que você usou no comando (o que tem pymc e nutpie). Rodar
todo dia não custa nada: sem jogos novos, ele encerra em segundos. A tarefa
roda quando você está logado; se o PC estiver desligado no horário, ela fica
para o dia seguinte. Para conferir: abra "Agendador de Tarefas" e procure
"Brasileirao - atualizar previsao" (a coluna "Último resultado" mostra 0x0
quando deu certo e 0x1 quando alguma conferência falhou).

RESULTADOS MANUAIS
------------------
Se o football-data ainda não trouxe a última rodada, digite os placares em
data/manual/resultados_recentes_2026.csv (mesmo formato das linhas que já
estão lá). Eles entram na simulação, mas sem odds: o Elo do mercado só usa
esses jogos quando o football-data os publicar.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent
PROCESSADOS = PASTA / "data" / "processed"
MANUAL = PASTA / "data" / "manual"
LOGS = PASTA / "logs"
MODELO_PAGINA = PASTA / "pagina" / "modelo_apresentacao.html"
PAGINA = PASTA / "apresentacao.html"          # fragmento, para a página publicada no Claude
PAGINA_SITE = PASTA / "docs" / "index.html"     # documento completo, para o GitHub Pages
BASE_GITHUB = PASTA / "data" / "raw" / "brasileirao_github_2003_2024.csv"

TEMPORADA = 2026
N_TIMES = 20
N_REBAIXADOS = 4
JOGOS_NA_TEMPORADA = N_TIMES * (N_TIMES - 1)          # 380
SUFIXO_BAYES = "_bayes_mercado_blocos4"                 # versão padrão do 05
HISTORICO = PROCESSADOS / "historico_previsoes.csv"
NOME_TAREFA = "Brasileirao - atualizar previsao"
ULTIMA = PROCESSADOS / "ultima_atualizacao.json"


class Falha(Exception):
    """Algo deu errado: a atualização para e nada é publicado."""


# =============================================================================
# 1. REGISTRO (tudo o que aparece na tela também vai para o arquivo de log)
# =============================================================================

class Registro:
    def __init__(self, caminho: Path):
        caminho.parent.mkdir(exist_ok=True)
        self.arquivo = open(caminho, "w", encoding="utf-8")
        self.caminho = caminho

    def __call__(self, texto: str = "") -> None:
        print(texto, flush=True)
        self.arquivo.write(texto + "\n")
        self.arquivo.flush()


log: Registro  # definido em main()


def rodar(script: str, *args: str, obrigatorio: bool = True) -> str:
    """Roda um dos scripts do projeto com o mesmo Python deste, mostrando e
    registrando a saída. Devolve o texto impresso.

    PYTHONUTF8=1 evita o erro clássico do Windows ao gravar acentos e símbolos
    (≤, −, ×) quando a saída vai para um arquivo em vez da tela."""
    log(f"\n$ python {script} {' '.join(args)}".rstrip())
    ambiente = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        # Pelo Agendador, o conda não é "ativado". Estas pastas são as que a
        # ativação acrescentaria ao PATH (DLLs do numpy, compilador do PyMC).
        base = Path(sys.prefix)
        extras = [base, base / "Library" / "mingw-w64" / "bin", base / "Library" / "usr" / "bin",
                  base / "Library" / "bin", base / "Scripts"]
        ambiente["PATH"] = os.pathsep.join([str(p) for p in extras if p.exists()] + [ambiente.get("PATH", "")])
    # -u: sem buffer. Sem ele, quando a saída vai para um "cano" (pipe), o Python
    # guarda os prints e só mostra no fim: parece que o script travou.
    processo = subprocess.Popen([sys.executable, "-u", script, *args], cwd=PASTA, env=ambiente,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
    linhas = []
    for linha in processo.stdout:
        linha = linha.rstrip("\n")
        linhas.append(linha)
        log("    " + linha)
    processo.wait()
    saida = "\n".join(linhas)
    if processo.returncode != 0:
        if processo.returncode in (3221225786, -2, 130):   # 0xC000013A = Ctrl+C no Windows
            raise KeyboardInterrupt
        mensagem = f"{script} terminou com erro (código {processo.returncode})"
        if obrigatorio:
            raise Falha(mensagem)
        log(f"  AVISO: {mensagem}; seguindo sem ele.")
    return saida


# =============================================================================
# 2. DADOS DE 2026: o que já se sabe
# =============================================================================

def jogos_conhecidos() -> tuple[pd.DataFrame, int]:
    """Todos os jogos de 2026 com placar: football-data + manuais que ainda não
    estão nele. Devolve também quantos jogos têm odds."""
    partidas = pd.read_csv(PROCESSADOS / "partidas.csv", parse_dates=["data"])
    fd = partidas[partidas["temporada"] == TEMPORADA].copy()
    fd["fonte"] = "football-data"
    n_odds = int(fd["prob_mercado_mandante"].notna().sum())

    caminho = MANUAL / f"resultados_recentes_{TEMPORADA}.csv"
    if not caminho.exists():
        return fd, n_odds
    manual = pd.read_csv(caminho, parse_dates=["data"])
    junto = manual.merge(fd[["mandante", "visitante", "gols_mandante", "gols_visitante"]],
                         on=["mandante", "visitante"], how="left", suffixes=("", "_fd"))
    ja_no_fd = junto["gols_mandante_fd"].notna()
    diferentes = ja_no_fd & ((junto["gols_mandante"] != junto["gols_mandante_fd"])
                             | (junto["gols_visitante"] != junto["gols_visitante_fd"]))
    if diferentes.any():
        raise Falha("Placar manual diferente do football-data (corrija o arquivo manual):\n"
                    + junto.loc[diferentes, ["mandante", "visitante", "gols_mandante", "gols_visitante",
                                             "gols_mandante_fd", "gols_visitante_fd"]].to_string(index=False))
    novos = manual.loc[~ja_no_fd.to_numpy(), ["data", "mandante", "visitante", "gols_mandante", "gols_visitante"]]
    novos = novos.assign(fonte="manual")
    if len(novos):
        log(f"  {len(novos)} resultados manuais ainda não estão no football-data (entram sem odds)")
    colunas = ["data", "mandante", "visitante", "gols_mandante", "gols_visitante", "fonte"]
    return pd.concat([fd[colunas], novos[colunas]], ignore_index=True), n_odds


def classificacao(jogos: pd.DataFrame) -> pd.DataFrame:
    """Tabela atual com os critérios de desempate do Brasileirão
    (pontos, vitórias, saldo, gols pró)."""
    lados = []
    for casa, fora in [("mandante", "visitante"), ("visitante", "mandante")]:
        gp, gc = jogos[f"gols_{casa}"], jogos[f"gols_{fora}"]
        lados.append(pd.DataFrame({"time": jogos[casa], "jogos": 1, "vitorias": (gp > gc).astype(int),
                                   "pontos": 3 * (gp > gc) + (gp == gc), "gols_pro": gp, "gols_contra": gc}))
    t = pd.concat(lados).groupby("time").sum()
    t["saldo"] = t["gols_pro"] - t["gols_contra"]
    t = t.sort_values(["pontos", "vitorias", "saldo", "gols_pro"], ascending=False)
    t["posicao"] = range(1, len(t) + 1)
    return t


def conferir_dados() -> dict:
    """As conferências que impedem publicar algo errado. Devolve um resumo."""
    log("\n=== Conferindo os dados de 2026 ===")
    jogos, n_odds = jogos_conhecidos()
    restantes = pd.read_csv(PROCESSADOS / f"jogos_restantes_{TEMPORADA}.csv")
    manual_pares = set(zip(jogos.loc[jogos["fonte"] == "manual", "mandante"],
                           jogos.loc[jogos["fonte"] == "manual", "visitante"]))
    a_jogar = restantes[[(m, v) not in manual_pares for m, v in zip(restantes["mandante"], restantes["visitante"])]]

    problemas = []
    times = set(jogos["mandante"]) | set(jogos["visitante"])
    if len(times) != N_TIMES:
        problemas.append(f"{len(times)} times em 2026 (esperado {N_TIMES}): {sorted(times)}")
    repetidos = jogos.duplicated(["mandante", "visitante"], keep=False)
    if repetidos.any():
        problemas.append("jogos repetidos:\n" + jogos.loc[repetidos].to_string(index=False))
    tabela = classificacao(jogos)
    if (tabela["jogos"] > N_TIMES * 2 - 2).any():
        problemas.append("time com mais de 38 jogos: " + str(tabela.index[tabela["jogos"] > 38].tolist()))
    total = len(jogos) + len(a_jogar)
    if total != JOGOS_NA_TEMPORADA:
        problemas.append(f"jogos conhecidos ({len(jogos)}) + a jogar ({len(a_jogar)}) = {total}, "
                         f"esperado {JOGOS_NA_TEMPORADA}")
    if problemas:
        raise Falha("Conferência dos dados falhou:\n  - " + "\n  - ".join(problemas))

    # "Rodada" = quantos jogos a maioria dos times já fez (ignora jogos adiados)
    rodada = int(tabela["jogos"].mode().max())
    ultimo = jogos["data"].max()
    log(f"  {len(jogos)} jogos conhecidos ({n_odds} com odds), {len(a_jogar)} a jogar; "
        f"dados até a {rodada}ª rodada (último jogo em {ultimo:%d/%m/%Y})")
    log("  20 times, sem repetições, ninguém com mais de 38 jogos, total 380: OK")
    return {"jogos_conhecidos": len(jogos), "jogos_com_odds": n_odds,
            "gols": int(jogos["gols_mandante"].sum() + jogos["gols_visitante"].sum()),
            "rodada": rodada, "ultimo_jogo": f"{ultimo:%Y-%m-%d}", "tabela": tabela,
            "n_jogos_total": int(len(pd.read_csv(PROCESSADOS / "partidas.csv")) + (jogos["fonte"] == "manual").sum())}


def tem_novidade(resumo: dict) -> bool:
    if not ULTIMA.exists():
        return True
    anterior = json.loads(ULTIMA.read_text(encoding="utf-8"))
    chave = ["jogos_conhecidos", "jogos_com_odds", "gols"]
    return any(anterior.get(k) != resumo[k] for k in chave)


# =============================================================================
# 3. MODELOS
# =============================================================================

def precisa_preparar_modelo() -> bool:
    """True na primeira vez ou se os arquivos do modelo são de antes do Elo do mercado."""
    if not (PROCESSADOS / "params_elo_mercado.json").exists():
        return True
    caminho = PROCESSADOS / "modelo_logistico.joblib"
    if not caminho.exists():
        return True
    import joblib
    return "elo_mercado_diff" not in joblib.load(caminho).get("colunas", [])


def conferir_simulacao(caminho: Path, nome: str) -> pd.DataFrame:
    sim = pd.read_csv(caminho, index_col="time")
    soma = sim["prob_rebaixamento"].sum()
    if len(sim) != N_TIMES or abs(soma - N_REBAIXADOS) > 0.01:
        raise Falha(f"{nome}: {len(sim)} times e soma das probabilidades de queda {soma:.3f} "
                    f"(esperado {N_TIMES} e {N_REBAIXADOS})")
    return sim


# =============================================================================
# 4. HISTÓRICO E PÁGINA
# =============================================================================

def atualizar_historico(resumo: dict, principal: pd.DataFrame, bayes: pd.DataFrame | None) -> pd.DataFrame:
    """Uma linha por time e rodada. Se a rodada já existe (ex.: rodou de novo
    depois de digitar um resultado manual), ela é substituída."""
    tabela = resumo["tabela"]
    novo = pd.DataFrame({
        "rodada": resumo["rodada"],
        "atualizado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "jogos_conhecidos": resumo["jogos_conhecidos"],
        "time": principal.index,
        "posicao": tabela.loc[principal.index, "posicao"].to_numpy(),
        "pontos": tabela.loc[principal.index, "pontos"].to_numpy(),
        "prob_principal": principal["prob_rebaixamento"].round(4).to_numpy(),
        "prob_bayes": (bayes["prob_rebaixamento"].reindex(principal.index).round(4).to_numpy()
                       if bayes is not None else float("nan")),
    })
    if HISTORICO.exists():
        antigo = pd.read_csv(HISTORICO)
        novo = pd.concat([antigo[antigo["rodada"] != resumo["rodada"]], novo], ignore_index=True)
    novo = novo.sort_values(["rodada", "prob_principal"], ascending=[True, False])
    novo.to_csv(HISTORICO, index=False)
    log(f"  histórico: {novo['rodada'].nunique()} rodadas em {HISTORICO.name}")
    return novo


def gerar_pagina(resumo: dict, principal: pd.DataFrame, bayes: pd.DataFrame | None,
                 historico: pd.DataFrame) -> None:
    """Coloca os números no modelo da página. O HTML continua um arquivo único
    (abre com dois cliques, funciona no GitHub Pages e como artifact)."""
    extras = json.loads((PROCESSADOS / f"simulacao_extras_{TEMPORADA}.json").read_text(encoding="utf-8"))
    diag_caminho = PROCESSADOS / f"diagnostico{SUFIXO_BAYES}_{TEMPORADA}.json"
    tabela = resumo["tabela"]

    def num(v, casas=1):
        return None if pd.isna(v) else round(float(v), casas)

    times = []
    for t, linha in principal.iterrows():
        times.append({
            "time": t, "pos": int(tabela.loc[t, "posicao"]), "pts": int(linha["pontos_hoje"]),
            "jr": int(linha["jogos_restantes"]),
            "p1": num(100 * linha["prob_rebaixamento"], 2),
            "pb": num(100 * bayes.loc[t, "prob_rebaixamento"], 2) if bayes is not None else None,
            "lo": num(linha["pontos_p10"], 0), "med": num(linha["pontos_finais_medio"]),
            "hi": num(linha["pontos_p90"], 0),
        })
    dados = {
        "temporada": TEMPORADA,
        "rodada": resumo["rodada"],
        "rodadas_restantes": int(principal["jogos_restantes"].max()),
        "ultimo_jogo": resumo["ultimo_jogo"],
        "atualizado_em": datetime.now().strftime("%Y-%m-%d"),
        "n_simulacoes": extras["n"],
        "n_jogos_total": resumo["n_jogos_total"],
        "times": times,
        "corte": extras["corte"],
        "salva_com": extras["salva_com"],
        "quartetos": extras["quartetos"],
        "bayes": json.loads(diag_caminho.read_text(encoding="utf-8")) if bayes is not None and diag_caminho.exists() else None,
        "historico": [{"r": int(h.rodada), "time": h.time,
                       "p1": num(100 * h.prob_principal, 2), "pb": num(100 * h.prob_bayes, 2)}
                      for h in historico.itertuples()],
    }
    texto = json.dumps(dados, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    modelo = MODELO_PAGINA.read_text(encoding="utf-8")
    if modelo.count("__DADOS__") != 1:
        raise Falha(f"{MODELO_PAGINA.name} precisa ter exatamente um __DADOS__")
    pagina = modelo.replace("__DADOS__", texto)
    PAGINA.write_text(pagina, encoding="utf-8")
    # O GitHub Pages precisa de um documento HTML completo (sem o doctype, o
    # navegador entra em "modo de compatibilidade" e o layout muda).
    corte = pagina.index("</style>") + len("</style>")
    site = ('<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body { margin: 0; } [hidden] { display: none !important; }</style>\n'
            + pagina[:corte] + "\n</head>\n<body>\n" + pagina[corte:] + "\n</body>\n</html>\n")
    PAGINA_SITE.parent.mkdir(exist_ok=True)
    PAGINA_SITE.write_text(site, encoding="utf-8")
    log(f"  página gerada: {PAGINA.name} e docs/index.html ({PAGINA.stat().st_size / 1024:.0f} KB)")


def ler_bayes() -> pd.DataFrame | None:
    caminho = PROCESSADOS / f"simulacao_{TEMPORADA}{SUFIXO_BAYES}.csv"
    return conferir_simulacao(caminho, "bayesiano") if caminho.exists() else None


def imprimir_resumo(principal: pd.DataFrame, bayes: pd.DataFrame | None, historico: pd.DataFrame,
                    rodada: int) -> None:
    anteriores = historico[historico["rodada"] < rodada]
    antes = (anteriores[anteriores["rodada"] == anteriores["rodada"].max()].set_index("time")["prob_principal"]
             if len(anteriores) else pd.Series(dtype=float))
    quadro = pd.DataFrame({
        "principal %": (100 * principal["prob_rebaixamento"]).round(1),
        "bayesiano %": (100 * bayes["prob_rebaixamento"].reindex(principal.index)).round(1) if bayes is not None else None,
        "variação (p.p.)": (100 * (principal["prob_rebaixamento"] - antes.reindex(principal.index))).round(1),
    })
    log(f"\n=== Chance de rebaixamento após a {rodada}ª rodada ===")
    log(quadro.head(10).to_string())


def publicar(rodada: int) -> None:
    """Commit + push dos arquivos que mudaram (o .gitignore decide quais).
    Com o GitHub Pages ligado na pasta docs/, a página pública se atualiza sozinha."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=PASTA, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    log("\n=== Publicando no GitHub ===")
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        log("  AVISO: esta pasta ainda não é um repositório git; nada foi publicado.")
        return
    git("add", "-A")
    if git("diff", "--cached", "--quiet").returncode == 0:
        log("  nada mudou desde o último commit")
        return
    r = git("commit", "-m", f"Atualização após a {rodada}ª rodada")
    log("  " + (r.stdout + r.stderr).strip().splitlines()[0])
    r = git("push")
    if r.returncode != 0:
        log("  AVISO: o push falhou (internet ou login do GitHub). O commit ficou salvo no PC;\n"
            "  ele sobe no próximo push. Detalhe: " + (r.stderr.strip().splitlines() or [""])[-1])
    else:
        log("  enviado; o GitHub Pages atualiza a página em 1 ou 2 minutos")


def agendar(hora: str | None, com_publicar: bool = False) -> int:
    """Cria (hora = "HH:MM") ou remove (hora = None) a tarefa diária no Windows."""
    if os.name != "nt":
        print("O agendamento automático só foi feito para o Windows (Agendador de Tarefas).")
        return 1
    if hora is None:
        cmd = ["schtasks", "/Delete", "/F", "/TN", NOME_TAREFA]
    else:
        acao = f'"{sys.executable}" "{PASTA / "atualizar.py"}"' + (" --publicar" if com_publicar else "")
        cmd = ["schtasks", "/Create", "/F", "/TN", NOME_TAREFA, "/SC", "DAILY", "/ST", hora, "/TR", acao]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    print((resultado.stdout + resultado.stderr).strip())
    if resultado.returncode == 0 and hora:
        print(f"Tarefa criada: todo dia às {hora}, com {sys.executable}")
        print(f"Os registros de cada execução ficam em {LOGS}")
    return resultado.returncode


# =============================================================================
# 5. PROGRAMA PRINCIPAL
# =============================================================================

def main() -> int:
    global log
    parser = argparse.ArgumentParser(description="Atualiza a previsão de ponta a ponta.")
    parser.add_argument("--forcar", action="store_true", help="roda mesmo sem jogos novos")
    parser.add_argument("--sem-download", action="store_true", help="usa o BRA.csv já baixado")
    parser.add_argument("--sem-bayes", action="store_true", help="pula o modelo bayesiano")
    parser.add_argument("--retreinar", action="store_true", help="treina de novo a logística (03)")
    parser.add_argument("--so-pagina", action="store_true", help="só gera a página com os arquivos atuais")
    parser.add_argument("--n", type=int, default=20000, help="simulações por modelo (padrão 20.000)")
    parser.add_argument("--publicar", action="store_true",
                        help="no fim, faz commit e push para o GitHub (atualiza o GitHub Pages)")
    parser.add_argument("--agendar", metavar="HH:MM", help="cria a tarefa diária no Agendador do Windows "
                        "(junto com --publicar, a tarefa também publica)")
    parser.add_argument("--desagendar", action="store_true", help="remove a tarefa do Agendador")
    args = parser.parse_args()
    if args.agendar or args.desagendar:
        return agendar(None if args.desagendar else args.agendar, args.publicar)

    inicio = datetime.now()
    log = Registro(LOGS / f"atualizacao_{inicio:%Y-%m-%d_%H%M}.log")
    log(f"=== Atualização de {inicio:%d/%m/%Y %H:%M} (Python {sys.version.split()[0]}) ===")
    try:
        if not args.so_pagina:
            if not args.sem_download:
                # A base do GitHub (2003–2024) só precisa ser baixada uma vez
                extra = [] if BASE_GITHUB.exists() else ["--conferir"]
                saida = rodar("01_coleta_dados.py", "--forcar-download", *extra)
                if "times não reconhecidos" in saida:
                    raise Falha("o football-data trouxe um nome de time desconhecido "
                                "(veja o AVISO acima e acrescente em APELIDOS no 01)")
            resumo = conferir_dados()
            if not (args.forcar or tem_novidade(resumo)):
                log("\nNenhum jogo novo desde a última atualização. Nada a fazer.")
                return 0

            if precisa_preparar_modelo():
                log("\nPrimeira execução (ou modelo de versão antiga): otimizando o Elo do mercado "
                    "e treinando a logística. Isso só acontece uma vez.")
                rodar("02_features.py", "--otimizar")
                rodar("03_modelo_logistico.py")
            else:
                rodar("02_features.py")
                if args.retreinar:
                    rodar("03_modelo_logistico.py")
            rodar("06_simulacao.py", "--n", str(args.n))
            if not args.sem_bayes:
                log("\nAjustando o bayesiano: costuma levar de 3 a 10 minutos, com alguns minutos "
                    "sem mostrar nada enquanto compila e amostra. Não feche a janela.")
                saida = rodar("05_modelo_bayesiano.py", "--n", str(args.n), obrigatorio=False)
                if "AVISO: o amostrador teve problemas" in saida:
                    log("  AVISO: o bayesiano não convergiu bem nesta rodada; confira o log antes de confiar nele.")
        else:
            resumo = conferir_dados()

        log("\n=== Conferindo as saídas e gerando a página ===")
        principal = conferir_simulacao(PROCESSADOS / f"simulacao_{TEMPORADA}.csv", "modelo principal")
        bayes = None if args.sem_bayes else ler_bayes()
        if bayes is not None and not args.so_pagina:
            # Não misturar um bayesiano de uma rodada anterior com o principal de agora
            arq = PROCESSADOS / f"simulacao_{TEMPORADA}{SUFIXO_BAYES}.csv"
            if datetime.fromtimestamp(arq.stat().st_mtime) < inicio:
                log("  AVISO: a simulação bayesiana é anterior a esta atualização; a página sai sem ela.")
                bayes = None
        log("  probabilidades somam 4 e há 20 times: OK")
        historico = atualizar_historico(resumo, principal, bayes)
        gerar_pagina(resumo, principal, bayes, historico)
        imprimir_resumo(principal, bayes, historico, resumo["rodada"])

        ULTIMA.write_text(json.dumps({k: v for k, v in resumo.items() if k != "tabela"}
                                     | {"atualizado_em": inicio.isoformat(timespec="minutes")},
                                     indent=2), encoding="utf-8")
        if args.publicar:
            publicar(resumo["rodada"])
        log(f"\nPronto em {(datetime.now() - inicio).seconds // 60} min. Registro: {log.caminho}")
        return 0
    except KeyboardInterrupt:
        log("\n*** ATUALIZAÇÃO INTERROMPIDA pelo usuário (Ctrl+C ou janela fechada).")
        log("    Nada foi publicado; a página anterior continua valendo.")
        return 130
    except Falha as erro:
        log(f"\n*** ATUALIZAÇÃO INTERROMPIDA: {erro}")
        log("    Nada foi publicado; a página anterior continua valendo.")
        log(f"    Registro completo: {log.caminho}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
