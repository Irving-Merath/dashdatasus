"""
Robô de coleta do Painel DATASUS
================================

Este arquivo é executado automaticamente pelo GitHub Actions.
Ele baixa os microdados públicos do DATASUS, resume (agrega) os dados
por município e ano, e salva arquivos pequenos na pasta `dados/`,
que o dashboard (app.py) lê.

Você NÃO precisa editar este arquivo. As únicas linhas que podem ser
ajustadas, se um dia quiser, estão no bloco "CONFIGURAÇÕES" logo abaixo.

Uso (feito pelo robô):
    python coletar.py --base sinasc --uf RJ   -> coleta uma base de um estado
    python coletar.py --consolidar            -> junta os resultados na pasta dados/
"""

import argparse
import datetime as dt
import ftplib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
ANO_INICIAL = 2010
ANO_FINAL = 2024

FTP_HOST = "ftp.datasus.gov.br"
PASTA_DADOS = Path("dados")      # onde ficam os resumos (lidos pelo dashboard)
PASTA_SAIDA = Path("saida")      # área temporária de cada robô

UFS = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}
COD_UF = {sigla: cod for cod, sigla in UFS.items()}


def aviso(msg):
    """Mostra um aviso amarelo na tela do GitHub Actions."""
    print(f"::warning::{msg}", flush=True)


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------
# LEITURA DE ARQUIVOS DBF (rápida e econômica em memória)
# ----------------------------------------------------------------------
def ler_dbf(caminho, colunas, bloco=400_000):
    """Lê só as colunas pedidas de um arquivo DBF, em blocos.

    Devolve DataFrames de texto (já sem espaços nas pontas).
    Colunas que não existem no arquivo vêm vazias.
    """
    with open(caminho, "rb") as f:
        cab = f.read(32)
        n_reg = int.from_bytes(cab[4:8], "little")
        tam_cab = int.from_bytes(cab[8:10], "little")
        tam_reg = int.from_bytes(cab[10:12], "little")
        campos = {}
        pos = 1  # o 1º byte de cada registro marca se ele foi apagado
        while True:
            d = f.read(32)
            if not d or d[0] == 0x0D:
                break
            nome = d[:11].split(b"\x00")[0].decode("latin-1").strip().upper()
            tam = d[16]
            campos[nome] = (pos, tam)
            pos += tam

    tamanho = os.path.getsize(caminho)
    n_reg = min(n_reg, (tamanho - tam_cab) // tam_reg)
    if n_reg <= 0:
        return
    mm = np.memmap(caminho, dtype=np.uint8, mode="r",
                   offset=tam_cab, shape=(n_reg, tam_reg))

    for ini in range(0, n_reg, bloco):
        fim = min(ini + bloco, n_reg)
        parte = mm[ini:fim]
        valido = parte[:, 0] != ord("*")
        df = {}
        for col in colunas:
            if col not in campos:
                df[col] = pd.Series([""] * int(valido.sum()), dtype="object")
                continue
            p, t = campos[col]
            bruto = np.ascontiguousarray(parte[valido, p:p + t]).view(f"S{t}").ravel()
            df[col] = pd.Series(bruto).str.decode("latin-1").str.strip()
        yield pd.DataFrame(df)


# ----------------------------------------------------------------------
# FTP DO DATASUS
# ----------------------------------------------------------------------
class FTPDatasus:
    def __init__(self):
        self.ftp = None
        self._cache_listas = {}

    def conectar(self):
        for tentativa in range(1, 6):
            try:
                self.ftp = ftplib.FTP(FTP_HOST, timeout=180)
                self.ftp.login()
                return
            except Exception as e:  # noqa: BLE001
                log(f"Falha ao conectar no FTP (tentativa {tentativa}): {e}")
                time.sleep(20 * tentativa)
        raise RuntimeError("Não foi possível conectar ao FTP do DATASUS.")

    def _garantir(self):
        if self.ftp is None:
            self.conectar()
        else:
            try:
                self.ftp.voidcmd("NOOP")
            except Exception:  # noqa: BLE001
                self.conectar()

    def listar(self, pasta):
        """Lista os arquivos de uma pasta: {NOME_EM_MAIUSCULO: nome_real}."""
        if pasta in self._cache_listas:
            return self._cache_listas[pasta]
        nomes = {}
        for tentativa in range(1, 4):
            try:
                self._garantir()
                nomes = {n.split("/")[-1].upper(): n.split("/")[-1]
                         for n in self.ftp.nlst(pasta)}
                break
            except ftplib.error_perm:
                nomes = {}  # pasta não existe
                break
            except Exception as e:  # noqa: BLE001
                log(f"Erro ao listar {pasta} (tentativa {tentativa}): {e}")
                self.ftp = None
                time.sleep(15 * tentativa)
        self._cache_listas[pasta] = nomes
        return nomes

    def tamanho(self, pasta, nome):
        try:
            self._garantir()
            return self.ftp.size(f"{pasta}/{nome}")
        except Exception:  # noqa: BLE001
            return None

    def baixar(self, pasta, nome, destino):
        for tentativa in range(1, 5):
            try:
                self._garantir()
                with open(destino, "wb") as f:
                    self.ftp.retrbinary(f"RETR {pasta}/{nome}", f.write,
                                        blocksize=1024 * 256)
                return True
            except Exception as e:  # noqa: BLE001
                log(f"Erro ao baixar {nome} (tentativa {tentativa}): {e}")
                self.ftp = None
                time.sleep(20 * tentativa)
        return False

    def encontrar(self, pastas, nome_arquivo):
        """Procura o arquivo nas pastas, na ordem. Devolve (pasta, nome, situacao)."""
        for pasta, situacao in pastas:
            lista = self.listar(pasta)
            if nome_arquivo.upper() in lista:
                return pasta, lista[nome_arquivo.upper()], situacao
        return None, None, None


def dbc_para_dbf(dbc, dbf):
    import pyreaddbc
    pyreaddbc.dbc2dbf(str(dbc), str(dbf))


# ----------------------------------------------------------------------
# FERRAMENTAS DE AGREGAÇÃO
# ----------------------------------------------------------------------
def mapear(serie, mapa, padrao="Ignorado"):
    return serie.map(mapa).fillna(padrao)


def faixas(valores, cortes, rotulos, padrao="Ignorado"):
    """Classifica números em faixas. Valores inválidos viram 'Ignorado'."""
    r = pd.cut(valores, bins=cortes, labels=rotulos, right=False)
    return r.astype("object").where(r.notna(), padrao)


def contar(df, coluna, nome_var):
    t = df.groupby(["cod_mun", coluna], observed=True).size().reset_index(name="n")
    t.columns = ["cod_mun", "categoria", "n"]
    t["variavel"] = nome_var
    t["categoria"] = t["categoria"].astype(str)
    return t


def somar_partes(partes):
    tudo = pd.concat(partes, ignore_index=True)
    return (tudo.groupby(["cod_mun", "variavel", "categoria"], observed=True)["n"]
                .sum().reset_index())


# ----------------------------------------------------------------------
# BASE: SINASC (nascidos vivos)
# ----------------------------------------------------------------------
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

SINASC_META = {
    "nome": "Nascimentos",
    "sigla": "SINASC",
    "descricao": "Nascidos vivos por município de residência da mãe "
                 "(Sistema de Informações sobre Nascidos Vivos).",
    "unidade": "nascidos vivos",
    "variaveis": {
        "sexo": {"rotulo": "Sexo do bebê", "tipo": "categorica",
                 "ordem": ["Masculino", "Feminino", "Ignorado"]},
        "raca_cor": {"rotulo": "Raça/cor do bebê", "tipo": "categorica",
                     "ordem": ["Branca", "Preta", "Amarela", "Parda", "Indígena", "Ignorado"]},
        "faixa_idade_mae": {"rotulo": "Idade da mãe (faixas)", "tipo": "categorica",
                            "ordem": ["Menos de 15", "15 a 19", "20 a 24", "25 a 29",
                                      "30 a 34", "35 a 39", "40 ou mais", "Ignorado"]},
        "escolaridade_mae": {"rotulo": "Escolaridade da mãe", "tipo": "categorica",
                             "ordem": ["Nenhuma", "1 a 3 anos", "4 a 7 anos",
                                       "8 a 11 anos", "12 anos ou mais", "Ignorado"]},
        "parto": {"rotulo": "Tipo de parto", "tipo": "categorica",
                  "ordem": ["Vaginal", "Cesáreo", "Ignorado"]},
        "gestacao": {"rotulo": "Semanas de gestação", "tipo": "categorica",
                     "ordem": ["Menos de 22", "22 a 27", "28 a 31", "32 a 36",
                               "37 a 41", "42 ou mais", "Ignorado"]},
        "faixa_peso": {"rotulo": "Peso ao nascer (faixas)", "tipo": "categorica",
                       "ordem": ["Menos de 1.500 g", "1.500 a 2.499 g",
                                 "2.500 a 3.999 g", "4.000 g ou mais", "Ignorado"]},
        "consultas": {"rotulo": "Consultas de pré-natal", "tipo": "categorica",
                      "ordem": ["Nenhuma", "1 a 3", "4 a 6", "7 ou mais", "Ignorado"]},
        "local_nascimento": {"rotulo": "Local do nascimento", "tipo": "categorica",
                             "ordem": ["Hospital", "Outro estabelecimento de saúde",
                                       "Domicílio", "Aldeia indígena", "Outros", "Ignorado"]},
        "mes": {"rotulo": "Mês do nascimento", "tipo": "mes", "ordem": MESES},
        "idade_mae": {"rotulo": "Idade da mãe", "tipo": "numerica", "unidade": "anos"},
        "peso": {"rotulo": "Peso ao nascer", "tipo": "numerica", "unidade": "g",
                 "observacao": "Valores arredondados para os 100 g mais próximos."},
    },
    "indicadores": [
        {"rotulo": "Partos cesáreos", "tipo": "pct", "variavel": "parto",
         "categorias": ["Cesáreo"]},
        {"rotulo": "Baixo peso (< 2.500 g)", "tipo": "pct", "variavel": "faixa_peso",
         "categorias": ["Menos de 1.500 g", "1.500 a 2.499 g"]},
        {"rotulo": "Prematuros (< 37 semanas)", "tipo": "pct", "variavel": "gestacao",
         "categorias": ["Menos de 22", "22 a 27", "28 a 31", "32 a 36"]},
        {"rotulo": "Mães com menos de 20 anos", "tipo": "pct", "variavel": "faixa_idade_mae",
         "categorias": ["Menos de 15", "15 a 19"]},
        {"rotulo": "7 ou mais consultas de pré-natal", "tipo": "pct", "variavel": "consultas",
         "categorias": ["7 ou mais"]},
        {"rotulo": "Idade média da mãe", "tipo": "media", "variavel": "idade_mae"},
        {"rotulo": "Peso médio ao nascer", "tipo": "media", "variavel": "peso"},
    ],
    "fonte": "Ministério da Saúde / DATASUS – SINASC",
}

SINASC_PASTAS = [
    ("/dissemin/publicos/SINASC/NOV/DNRES", "final"),
    ("/dissemin/publicos/SINASC/1996_/Dados/DNRES", "final"),
    ("/dissemin/publicos/SINASC/PRELIM/DNRES", "preliminar"),
]


def sinasc_arquivos(uf, ano):
    return [f"DN{uf}{ano}.DBC"]


def sinasc_agregar(caminho_dbf, uf):
    cols = ["CODMUNRES", "SEXO", "RACACOR", "IDADEMAE", "ESCMAE", "PARTO",
            "GESTACAO", "PESO", "CONSULTAS", "LOCNASC", "DTNASC"]
    partes = []
    for b in ler_dbf(caminho_dbf, cols):
        df = pd.DataFrame({"cod_mun": b["CODMUNRES"].str[:6]})
        df.loc[~df["cod_mun"].str.fullmatch(r"\d{6}"), "cod_mun"] = COD_UF[uf] + "0000"
        idade = pd.to_numeric(b["IDADEMAE"], errors="coerce")
        idade = idade.where((idade >= 10) & (idade <= 60))
        peso = pd.to_numeric(b["PESO"], errors="coerce")
        peso = peso.where((peso >= 200) & (peso <= 7000))
        mes = pd.to_numeric(b["DTNASC"].str[2:4], errors="coerce")

        df["total"] = "Total"
        df["sexo"] = mapear(b["SEXO"], {"1": "Masculino", "M": "Masculino",
                                        "2": "Feminino", "F": "Feminino"})
        df["raca_cor"] = mapear(b["RACACOR"], {"1": "Branca", "2": "Preta", "3": "Amarela",
                                               "4": "Parda", "5": "Indígena"})
        df["faixa_idade_mae"] = faixas(idade, [10, 15, 20, 25, 30, 35, 40, 61],
                                       ["Menos de 15", "15 a 19", "20 a 24", "25 a 29",
                                        "30 a 34", "35 a 39", "40 ou mais"])
        df["escolaridade_mae"] = mapear(b["ESCMAE"], {"1": "Nenhuma", "2": "1 a 3 anos",
                                                      "3": "4 a 7 anos", "4": "8 a 11 anos",
                                                      "5": "12 anos ou mais"})
        df["parto"] = mapear(b["PARTO"], {"1": "Vaginal", "2": "Cesáreo"})
        df["gestacao"] = mapear(b["GESTACAO"], {"1": "Menos de 22", "2": "22 a 27",
                                                "3": "28 a 31", "4": "32 a 36",
                                                "5": "37 a 41", "6": "42 ou mais"})
        df["faixa_peso"] = faixas(peso, [0, 1500, 2500, 4000, 10000],
                                  ["Menos de 1.500 g", "1.500 a 2.499 g",
                                   "2.500 a 3.999 g", "4.000 g ou mais"])
        df["consultas"] = mapear(b["CONSULTAS"], {"1": "Nenhuma", "2": "1 a 3",
                                                  "3": "4 a 6", "4": "7 ou mais"})
        df["local_nascimento"] = mapear(b["LOCNASC"], {
            "1": "Hospital", "2": "Outro estabelecimento de saúde", "3": "Domicílio",
            "4": "Outros", "5": "Aldeia indígena"})
        df["mes"] = mes.map(lambda m: MESES[int(m) - 1] if m == m and 1 <= m <= 12 else None)
        df["idade_mae"] = idade.astype("Int64")
        df["peso"] = ((peso / 100).round() * 100).astype("Int64")

        for var in ["total", "sexo", "raca_cor", "faixa_idade_mae", "escolaridade_mae",
                    "parto", "gestacao", "faixa_peso", "consultas", "local_nascimento",
                    "mes", "idade_mae", "peso"]:
            sub = df[["cod_mun", var]].dropna()
            partes.append(contar(sub, var, var))
    return somar_partes(partes)


# ----------------------------------------------------------------------
# CATÁLOGO DE BASES
# Bases com "ativa": False aparecem no dashboard como "em breve".
# Elas serão ativadas uma por vez, nas próximas etapas.
# ----------------------------------------------------------------------
BASES = {
    "sinasc": {"ativa": True, "meta": SINASC_META, "pastas": SINASC_PASTAS,
               "arquivos": sinasc_arquivos, "agregar": sinasc_agregar},
    "sim": {"ativa": False, "meta": {"nome": "Óbitos", "sigla": "SIM"}},
    "sih": {"ativa": False, "meta": {"nome": "Internações hospitalares", "sigla": "SIH"}},
    "sinan": {"ativa": False, "meta": {"nome": "Doenças de notificação", "sigla": "SINAN"}},
    "cnes": {"ativa": False, "meta": {"nome": "Estabelecimentos de saúde", "sigla": "CNES"}},
    "sia": {"ativa": False, "meta": {"nome": "Atendimentos ambulatoriais", "sigla": "SIA"}},
    "pni": {"ativa": False, "meta": {"nome": "Vacinação", "sigla": "PNI"}},
}


# ----------------------------------------------------------------------
# COLETA DE UM ESTADO
# ----------------------------------------------------------------------
def ler_json(caminho, padrao):
    try:
        return json.loads(Path(caminho).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return padrao


def coletar(base, uf):
    cfg = BASES[base]
    if not cfg.get("ativa"):
        aviso(f"A base '{base}' ainda não foi ativada. Nada a fazer.")
        return 0

    controle_atual = PASTA_DADOS / base / "_controle" / f"{uf}.json"
    controle = ler_json(controle_atual, {})
    saida = PASTA_SAIDA / base
    (saida / "_uf").mkdir(parents=True, exist_ok=True)
    (saida / "_controle").mkdir(parents=True, exist_ok=True)

    ftp = FTPDatasus()
    falhas, novos = [], 0
    tmp = Path(tempfile.mkdtemp())

    for ano in range(ANO_INICIAL, ANO_FINAL + 1):
        for nome in cfg["arquivos"](uf, ano):
            pasta, real, situacao = ftp.encontrar(cfg["pastas"], nome)
            if not pasta:
                log(f"{nome}: não encontrado no DATASUS (talvez ainda não publicado).")
                continue
            tam = ftp.tamanho(pasta, real)
            chave = f"{ano}"
            assinatura = f"{pasta}/{real}:{tam}"
            arq_final = PASTA_DADOS / base / f"{uf}_{ano}.parquet"
            if controle.get(chave) == assinatura and arq_final.exists():
                log(f"{nome}: sem mudanças desde a última coleta. Pulando.")
                continue

            log(f"{nome}: baixando ({situacao}, {tam or '?'} bytes)...")
            dbc, dbf = tmp / real, tmp / (Path(real).stem + ".dbf")
            if not ftp.baixar(pasta, real, dbc):
                falhas.append(nome)
                continue
            try:
                dbc_para_dbf(dbc, dbf)
                res = cfg["agregar"](dbf, uf)
            except Exception as e:  # noqa: BLE001
                aviso(f"{nome}: erro ao processar ({e}).")
                falhas.append(nome)
                continue
            finally:
                for p in (dbc, dbf):
                    p.unlink(missing_ok=True)

            res.insert(0, "ano", ano)
            res["n"] = res["n"].astype("int32")
            sufixo = "" if situacao == "final" else "_prelim"
            res.to_parquet(saida / f"{uf}_{ano}{sufixo}.parquet", index=False)

            # versão resumida por estado (usada na visão "Brasil")
            r_uf = (res.groupby(["ano", "variavel", "categoria"])["n"].sum()
                       .reset_index())
            r_uf.insert(1, "cod_mun", COD_UF[uf])
            r_uf.to_parquet(saida / "_uf" / f"{uf}_{ano}{sufixo}.parquet", index=False)

            controle[chave] = assinatura
            novos += 1
            log(f"{nome}: ok ({int(res.loc[res.variavel == 'total', 'n'].sum()):,} registros).")

    (saida / "_controle" / f"{uf}.json").write_text(
        json.dumps(controle, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"Concluído {base}/{uf}: {novos} arquivo(s) novo(s), {len(falhas)} falha(s).")
    if falhas:
        aviso(f"{base}/{uf}: não foi possível processar {', '.join(falhas)}. "
              "O robô tentará de novo na próxima execução.")
    return 0


# ----------------------------------------------------------------------
# CONSOLIDAÇÃO (junta o trabalho de todos os robôs)
# ----------------------------------------------------------------------
def atualizar_municipios():
    """Baixa os nomes dos municípios do IBGE (só se ainda não tiver)."""
    destino = PASTA_DADOS / "municipios.csv"
    if destino.exists():
        return
    try:
        import requests
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        dados = requests.get(url, timeout=60).json()
        linhas = [{"cod_mun": str(m["id"])[:6], "municipio": m["nome"],
                   "uf": UFS.get(str(m["id"])[:2], "")} for m in dados]
        pd.DataFrame(linhas).sort_values("cod_mun").to_csv(destino, index=False)
        log(f"Lista de municípios salva ({len(linhas)}).")
    except Exception as e:  # noqa: BLE001
        aviso(f"Não foi possível baixar os nomes dos municípios do IBGE: {e}")


def consolidar():
    PASTA_DADOS.mkdir(exist_ok=True)
    if PASTA_SAIDA.exists():
        for arq in PASTA_SAIDA.rglob("*"):
            if arq.is_file():
                destino = PASTA_DADOS / arq.relative_to(PASTA_SAIDA)
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arq, destino)
                # um arquivo novo substitui a versão "irmã" (final x preliminar)
                if arq.suffix == ".parquet":
                    stem = destino.stem
                    irmao = (stem[:-7] if stem.endswith("_prelim") else stem + "_prelim")
                    (destino.parent / f"{irmao}.parquet").unlink(missing_ok=True)

    for base, cfg in BASES.items():
        pasta = PASTA_DADOS / base
        if cfg.get("ativa") and pasta.exists() and any(pasta.glob("*.parquet")):
            meta = dict(cfg["meta"])
            meta["atualizado_em"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
            (pasta / "_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    atualizar_municipios()
    log("Consolidação concluída.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--uf")
    ap.add_argument("--consolidar", action="store_true")
    a = ap.parse_args()
    if a.consolidar:
        consolidar()
    elif a.base and a.uf:
        sys.exit(coletar(a.base.lower(), a.uf.upper()))
    else:
        ap.print_help()
