"""
Painel DATASUS – dashboard em Streamlit
=======================================

Lê os resumos gerados pelo robô (pasta `dados/`) e mostra estatísticas
descritivas com gráficos. Novas bases aparecem sozinhas no menu "Tema"
assim que o robô coleta os dados delas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Painel de dados do SUS", page_icon="📊", layout="wide")

PASTA = Path(__file__).parent / "dados"

CATALOGO = {  # todas as bases previstas, na ordem do menu
    "sinasc": "Nascimentos (SINASC)",
    "sim": "Óbitos (SIM)",
    "sih": "Internações hospitalares (SIH)",
    "sinan": "Doenças de notificação (SINAN)",
    "cnes": "Estabelecimentos de saúde (CNES)",
    "sia": "Atendimentos ambulatoriais (SIA)",
    "pni": "Vacinação (PNI)",
}
UFS = {
    "11": ("RO", "Rondônia"), "12": ("AC", "Acre"), "13": ("AM", "Amazonas"),
    "14": ("RR", "Roraima"), "15": ("PA", "Pará"), "16": ("AP", "Amapá"),
    "17": ("TO", "Tocantins"), "21": ("MA", "Maranhão"), "22": ("PI", "Piauí"),
    "23": ("CE", "Ceará"), "24": ("RN", "Rio Grande do Norte"), "25": ("PB", "Paraíba"),
    "26": ("PE", "Pernambuco"), "27": ("AL", "Alagoas"), "28": ("SE", "Sergipe"),
    "29": ("BA", "Bahia"), "31": ("MG", "Minas Gerais"), "32": ("ES", "Espírito Santo"),
    "33": ("RJ", "Rio de Janeiro"), "35": ("SP", "São Paulo"), "41": ("PR", "Paraná"),
    "42": ("SC", "Santa Catarina"), "43": ("RS", "Rio Grande do Sul"),
    "50": ("MS", "Mato Grosso do Sul"), "51": ("MT", "Mato Grosso"),
    "52": ("GO", "Goiás"), "53": ("DF", "Distrito Federal"),
}
SIGLA_COD = {s: c for c, (s, _) in UFS.items()}
CORES = ["#0B7A75", "#E0A526", "#2F5D8A", "#9C6644", "#7FB7B2", "#6B5B95", "#B8B8B8"]
COR_PRINCIPAL = CORES[0]
CONFIG_GRAF = {"displayModeBar": False, "locale": "pt-BR"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap');
    html, body, [class*="st-"], .stMarkdown, .stText, button, input, select, textarea {
        font-family: 'Public Sans', 'Segoe UI', Roboto, Arial, sans-serif;
    }
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1 { font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .sub { opacity: 0.75; margin-top: 0; margin-bottom: 1.2rem; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem; margin: 0.5rem 0 1.25rem; }
    .kpi { border-left: 4px solid #0B7A75; padding: 0.4rem 0.8rem; }
    .kpi .valor { font-size: 1.6rem; font-weight: 800; line-height: 1.15;
                  font-variant-numeric: tabular-nums; }
    .kpi .rotulo { font-size: 0.85rem; opacity: 0.75; }
    .kpi.destaque { border-left-color: #E0A526; }
    .kpi.destaque .valor { font-size: 2.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Formatação em português
# ----------------------------------------------------------------------
def fmt_int(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_dec(x, casas=1):
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_unidade(x, un):
    """Número com a unidade: 'R$ 1.234,56', '3.250 g', '27,4 anos'."""
    if un == "R$":
        for limite, nome in ((1e9, "bilhões"), (1e6, "milhões")):
            if abs(x) >= limite:
                return f"R$ {fmt_dec(x / limite, 1)} {nome}"
        return f"R$ {fmt_dec(x, 2)}"
    casas = 0 if un == "g" else 1
    return f"{fmt_dec(x, casas)} {un}".strip()


# ----------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------
def versao_dados():
    """Assinatura dos dados: muda sempre que o robô salva algo novo.

    Ela é passada para as funções abaixo; quando muda, o dashboard
    descarta a memória (cache) e relê os arquivos na hora.
    """
    arqs = [PASTA / "municipios.csv"] + [PASTA / b / "_meta.json" for b in CATALOGO]
    return tuple(a.stat().st_mtime if a.exists() else 0 for a in arqs)


@st.cache_data(ttl=3600, show_spinner=False)
def ler_metas(versao):
    metas = {}
    for base in CATALOGO:
        arq = PASTA / base / "_meta.json"
        if arq.exists():
            metas[base] = json.loads(arq.read_text(encoding="utf-8"))
    return metas


@st.cache_data(ttl=3600, show_spinner=False)
def listar_arquivos(base, versao):
    """Devolve uma tabela com uf, ano, preliminar e caminho de cada arquivo."""
    linhas = []
    for nivel, pasta in (("mun", PASTA / base), ("uf", PASTA / base / "_uf")):
        for arq in pasta.glob("*.parquet"):
            partes = arq.stem.split("_")
            if len(partes) < 2 or not partes[1].isdigit():
                continue
            linhas.append({"nivel": nivel, "uf": partes[0], "ano": int(partes[1]),
                           "prelim": arq.stem.endswith("_prelim"), "caminho": str(arq)})
    return pd.DataFrame(linhas, columns=["nivel", "uf", "ano", "prelim", "caminho"])


@st.cache_data(ttl=3600, show_spinner="Carregando dados…")
def carregar(base, nivel, uf, anos, versao, agravo=None):
    arqs = listar_arquivos(base, versao)
    sel = arqs[(arqs.nivel == nivel) & arqs.ano.between(anos[0], anos[1])]
    if uf:
        sel = sel[sel.uf == uf]
    if sel.empty:
        return pd.DataFrame(columns=["ano", "cod_mun", "variavel", "categoria", "n"])
    df = pd.concat([pd.read_parquet(c) for c in sel.caminho], ignore_index=True)
    if agravo:  # SINAN: as variáveis vêm como "DENG|sexo"; fica só a doença escolhida
        prefixo = f"{agravo}|"
        df = df[df.variavel.astype(str).str.startswith(prefixo)].copy()
        df["variavel"] = df.variavel.astype(str).str[len(prefixo):]
    for col in ("cod_mun", "variavel", "categoria"):
        df[col] = df[col].astype("category")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def municipios(versao):
    arq = PASTA / "municipios.csv"
    if arq.exists():
        m = pd.read_csv(arq, dtype=str)
        return dict(zip(m.cod_mun, m.municipio))
    return {}


def nome_local(cod, nomes_mun):
    cod = str(cod)
    if len(cod) == 2:
        return UFS.get(cod, (cod, cod))[1]
    if cod.endswith("0000"):
        return "Município ignorado"
    return nomes_mun.get(cod, cod)


# ----------------------------------------------------------------------
# Estatística descritiva
# ----------------------------------------------------------------------
def total(df):
    return int(df.loc[df.variavel == "total", "n"].sum())


def proporcao(df, var, cats):
    sub = df[df.variavel == var]
    den = sub.n.sum()
    return (sub[sub.categoria.isin(cats)].n.sum() / den * 100) if den else np.nan


def valores_numericos(df, var):
    sub = df[df.variavel == var]
    if sub.empty:
        return pd.Series(dtype=float)
    s = sub.groupby("categoria", observed=True).n.sum()
    s.index = pd.to_numeric(s.index.astype(str), errors="coerce")
    return s[s.index.notna()].sort_index()


def resumo_numerico(cont):
    """Estatísticas a partir de uma tabela de frequências (valor -> contagem)."""
    if cont.empty or cont.sum() == 0:
        return None
    v, n = cont.index.to_numpy(float), cont.to_numpy(float)
    N = n.sum()
    media = (v * n).sum() / N
    dp = np.sqrt(((v - media) ** 2 * n).sum() / max(N - 1, 1))
    acum = np.cumsum(n) / N

    def q(p):
        return v[np.searchsorted(acum, p)]

    return {"N": N, "Média": media, "Desvio padrão": dp, "Mínimo": v.min(),
            "1º quartil": q(0.25), "Mediana": q(0.5), "3º quartil": q(0.75),
            "Máximo": v.max(), "Moda": v[np.argmax(n)]}


def media(df, var):
    r = resumo_numerico(valores_numericos(df, var))
    return r["Média"] if r else np.nan


def valor_indicador(df, ind):
    if ind["tipo"] == "pct":
        return proporcao(df, ind["variavel"], ind["categorias"])
    if ind["tipo"] == "media":
        return media(df, ind["variavel"])
    if ind["tipo"] == "soma":
        cont = valores_numericos(df, ind["variavel"])
        return float((cont.index.to_numpy(float) * cont.to_numpy(float)).sum())
    return total(df)


def formatar_indicador(valor, ind, meta):
    if pd.isna(valor):
        return "–"
    if ind["tipo"] == "pct":
        return f"{fmt_dec(valor)}%"
    if ind["tipo"] in ("media", "soma"):
        un = meta["variaveis"].get(ind["variavel"], {}).get("unidade", "")
        return fmt_unidade(valor, un)
    return fmt_int(valor)


def estilo(fig, altura=380):
    fig.update_layout(height=altura, margin=dict(l=8, r=8, t=30, b=8),
                      font=dict(family="Public Sans, Arial, sans-serif"),
                      legend=dict(orientation="h", y=-0.2), separators=",.")
    return fig


# ----------------------------------------------------------------------
# Cabeçalho e filtros
# ----------------------------------------------------------------------
st.title("Painel de dados do SUS")
st.markdown('<p class="sub">Estatísticas descritivas com dados públicos do DATASUS, '
            'por estado e município.</p>', unsafe_allow_html=True)

VERSAO = versao_dados()
metas = ler_metas(VERSAO)
opcoes = list(CATALOGO)
rotulos = {b: CATALOGO[b] + ("" if b in metas else " · em breve") for b in opcoes}

c1, c2 = st.columns([1, 1])
base = c1.selectbox("Tema", opcoes, format_func=lambda b: rotulos[b])

if base not in metas:
    st.info(f"**{CATALOGO[base]}** ainda não está disponível. Esta base será ativada "
            "nas próximas etapas do projeto. Escolha outro tema para continuar.")
    st.stop()

meta = metas[base]
variaveis = meta["variaveis"]
indicadores = meta.get("indicadores", [])
nome_tema = meta["nome"]
agravos = meta.get("agravos")  # só o SINAN tem: várias doenças na mesma base
agravo = None
if agravos:
    agravo = c1.selectbox("Doença ou agravo", list(agravos),
                          format_func=lambda a: agravos[a]["nome"])
    variaveis = {k: variaveis[k] for k in agravos[agravo]["variaveis"] if k in variaveis}
    indicadores = agravos[agravo]["indicadores"]
    nome_tema = agravos[agravo]["nome"]

arqs = listar_arquivos(base, VERSAO)
if arqs.empty:
    st.warning("Os dados desta base ainda estão sendo coletados. Volte mais tarde.")
    st.stop()

anos_disp = sorted(arqs.ano.unique())
if len(anos_disp) > 1:
    anos = c2.select_slider("Período", options=anos_disp,
                            value=(anos_disp[0], anos_disp[-1]))
else:
    anos = (anos_disp[0], anos_disp[0])
    c2.markdown(f"**Período:** {anos_disp[0]}")

c3, c4 = st.columns([1, 1])
ufs_disp = sorted(arqs[arqs.nivel == "mun"].uf.unique(), key=lambda s: UFS[SIGLA_COD[s]][1])
local = c3.selectbox("Local", ["BR"] + ufs_disp,
                     format_func=lambda s: "Brasil (todos os estados)" if s == "BR"
                     else UFS[SIGLA_COD[s]][1])

nomes_mun = municipios(VERSAO)
if local == "BR":
    df = carregar(base, "uf", None, anos, VERSAO, agravo)
    c4.selectbox("Municípios", ["Escolha um estado para filtrar municípios"], disabled=True)
    titulo_local = "Brasil"
else:
    df_uf = carregar(base, "mun", local, anos, VERSAO, agravo)
    cods = sorted(df_uf.cod_mun.unique(), key=lambda c: nome_local(c, nomes_mun))
    escolha = c4.multiselect("Municípios", cods, format_func=lambda c: nome_local(c, nomes_mun),
                             placeholder="Todos os municípios do estado")
    df = df_uf[df_uf.cod_mun.isin(escolha)] if escolha else df_uf
    titulo_local = UFS[SIGLA_COD[local]][1] if not escolha else (
        nome_local(escolha[0], nomes_mun) if len(escolha) == 1
        else f"{len(escolha)} municípios de {local}")

periodo_txt = str(anos[0]) if anos[0] == anos[1] else f"{anos[0]} a {anos[1]}"
prelim = sorted(arqs[(arqs.prelim) & arqs.ano.between(*anos)].ano.unique())
if prelim:
    st.warning("Dados preliminares (podem mudar) para: "
               + ", ".join(str(a) for a in prelim) + ".")

if df.empty or total(df) == 0:
    st.info("Não há registros para essa combinação de filtros.")
    st.stop()

aba1, aba2, aba3, aba4 = st.tabs(["Visão geral", "Distribuições", "Comparar locais", "Tabela"])

# ----------------------------------------------------------------------
# Aba 1 – Visão geral
# ----------------------------------------------------------------------
with aba1:
    st.subheader(f"{nome_tema} · {titulo_local} · {periodo_txt}")
    cards = [f'<div class="kpi destaque"><div class="valor">{fmt_int(total(df))}</div>'
             f'<div class="rotulo">{meta["unidade"].capitalize()}</div></div>']
    for ind in indicadores:
        v = formatar_indicador(valor_indicador(df, ind), ind, meta)
        cards.append(f'<div class="kpi"><div class="valor">{v}</div>'
                     f'<div class="rotulo">{ind["rotulo"]}</div></div>')
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)

    por_ano = df[df.variavel == "total"].groupby("ano").n.sum().reset_index()
    if len(por_ano) > 1:
        fig = px.line(por_ano, x="ano", y="n", markers=True,
                      labels={"ano": "Ano", "n": meta["unidade"].capitalize()},
                      color_discrete_sequence=[COR_PRINCIPAL])
        fig.update_xaxes(dtick=1)
        st.markdown(f"**Evolução anual de {meta['unidade']}**")
        st.plotly_chart(estilo(fig), config=CONFIG_GRAF)

    if "mes" in variaveis and (df.variavel == "mes").any():
        ordem = variaveis["mes"]["ordem"]
        mes = (df[df.variavel == "mes"].groupby("categoria", observed=True).n.sum()
               .reindex(ordem).fillna(0).reset_index())
        mes.columns = ["Mês", "n"]
        mes["n"] = mes["n"] / (anos[1] - anos[0] + 1)
        fig = px.bar(mes, x="Mês", y="n", labels={"n": "Média por ano"},
                     color_discrete_sequence=[CORES[2]])
        st.markdown("**Distribuição por mês** (média anual no período)")
        st.plotly_chart(estilo(fig, 320), config=CONFIG_GRAF)

# ----------------------------------------------------------------------
# Aba 2 – Distribuições
# ----------------------------------------------------------------------
with aba2:
    nomes_var = [k for k, v in variaveis.items() if v["tipo"] in ("categorica", "numerica")]
    var = st.selectbox("Variável", nomes_var, format_func=lambda k: variaveis[k]["rotulo"])
    info = variaveis[var]

    if info["tipo"] == "categorica":
        sub = df[df.variavel == var]
        dist = (sub.groupby("categoria", observed=True).n.sum()
                .reindex(info.get("ordem")).dropna().reset_index())
        dist.columns = ["Categoria", "n"]
        dist["%"] = dist.n / dist.n.sum() * 100
        dist["rotulo"] = dist.apply(lambda r: f"{fmt_dec(r['%'])}% ({fmt_int(r.n)})", axis=1)
        fig = px.bar(dist, y="Categoria", x="%", orientation="h", text="rotulo",
                     color_discrete_sequence=[COR_PRINCIPAL])
        fig.update_yaxes(categoryorder="array", categoryarray=dist.Categoria[::-1], title=None)
        fig.update_xaxes(title="% do total", range=[0, dist["%"].max() * 1.35])
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(estilo(fig, 60 + 45 * len(dist)), config=CONFIG_GRAF)

        if anos[0] != anos[1] and st.toggle("Ver evolução por ano"):
            ev = sub.groupby(["ano", "categoria"], observed=True).n.sum().reset_index()
            ev["%"] = ev.n / ev.groupby("ano").n.transform("sum") * 100
            fig = px.bar(ev, x="ano", y="%", color="categoria",
                         category_orders={"categoria": info.get("ordem")},
                         color_discrete_sequence=CORES,
                         labels={"ano": "Ano", "categoria": ""})
            fig.update_xaxes(dtick=1)
            st.plotly_chart(estilo(fig, 420), config=CONFIG_GRAF)
        st.caption("Percentuais calculados sobre o total, incluindo registros sem "
                   "informação (\"Ignorado\").")
    else:
        cont = valores_numericos(df, var)
        r = resumo_numerico(cont)
        un = info.get("unidade", "")
        if r is None:
            st.info("Sem valores válidos para essa variável.")
        else:
            fig = go.Figure(go.Bar(x=cont.index, y=cont.values, marker_color=COR_PRINCIPAL))
            for nome, cor, traco in (("Média", CORES[1], "dash"), ("Mediana", CORES[2], "dot")):
                fig.add_vline(x=r[nome], line_color=cor, line_dash=traco,
                              annotation_text=f"{nome}: {fmt_unidade(r[nome], un)}")
            fig.update_xaxes(title=f"{info['rotulo']} ({un})")
            fig.update_yaxes(title="Quantidade")
            st.plotly_chart(estilo(fig), config=CONFIG_GRAF)
            tab = pd.DataFrame({
                "Estatística": list(r.keys()),
                "Valor": [fmt_int(r["N"])] + [fmt_unidade(v, un) for k, v in r.items()
                                              if k != "N"],
            })
            tab.loc[0, "Estatística"] = "Registros com valor válido"
            st.dataframe(tab, hide_index=True)
            if info.get("observacao"):
                st.caption(info["observacao"])

# ----------------------------------------------------------------------
# Aba 3 – Comparar locais
# ----------------------------------------------------------------------
with aba3:
    inds = [{"rotulo": meta["unidade"].capitalize(), "tipo": "total"}] + indicadores
    ind = st.selectbox("Indicador", inds, format_func=lambda i: i["rotulo"])
    grupos = []
    for cod, g in df.groupby("cod_mun", observed=True):
        if g.empty:
            continue
        grupos.append({"Local": nome_local(cod, nomes_mun), "valor": valor_indicador(g, ind),
                       "n": total(g)})
    comp = pd.DataFrame(grupos).dropna(subset=["valor"])
    if comp.empty:
        st.info("Sem dados para comparar.")
    else:
        comp = comp.sort_values("valor", ascending=False)
        if ind["tipo"] != "total":
            minimo = st.number_input(f"Mostrar só locais com pelo menos quantos {meta['unidade']}?",
                                     min_value=0, value=100, step=50,
                                     help="Locais com poucos registros geram percentuais instáveis.")
            comp = comp[comp.n >= minimo]
        maximo = len(comp)
        qtd = st.slider("Quantos locais mostrar", 1, maximo, min(15, maximo)) if maximo > 1 else 1
        mostrar = comp.head(qtd).copy()
        mostrar["texto"] = [formatar_indicador(v, ind, meta) for v in mostrar.valor]
        fig = px.bar(mostrar, y="Local", x="valor", orientation="h", text="texto",
                     hover_data={"n": True, "valor": False},
                     color_discrete_sequence=[CORES[2]])
        fig.update_yaxes(categoryorder="total ascending", title=None)
        fig.update_xaxes(title=ind["rotulo"], range=[0, mostrar.valor.max() * 1.3])
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(estilo(fig, 80 + 32 * len(mostrar)), config=CONFIG_GRAF)

# ----------------------------------------------------------------------
# Aba 4 – Tabela
# ----------------------------------------------------------------------
with aba4:
    nomes_tab = [k for k, v in variaveis.items() if v["tipo"] in ("categorica", "mes")]
    var_t = st.selectbox("Variável da tabela", nomes_tab,
                         format_func=lambda k: variaveis[k]["rotulo"], key="tab")
    piv = (df[df.variavel == var_t].pivot_table(index="categoria", columns="ano", values="n",
                                                aggfunc="sum", observed=True)
           .reindex(variaveis[var_t].get("ordem")).dropna(how="all").fillna(0).astype(int))
    piv.columns = [str(c) for c in piv.columns]
    piv["Total"] = piv.sum(axis=1)
    piv.index.name = variaveis[var_t]["rotulo"]
    st.dataframe(piv)
    st.download_button("Baixar tabela (CSV)", piv.to_csv(sep=";", encoding="utf-8-sig"),
                       file_name=f"{base}_{var_t}_{local}_{periodo_txt}.csv".replace(" ", "_"),
                       mime="text/csv")

st.divider()
st.caption(f"Fonte: {meta.get('fonte', 'DATASUS')}. {meta.get('descricao', '')} "
           f"Dados atualizados pelo robô em {meta.get('atualizado_em', '–')}.")
