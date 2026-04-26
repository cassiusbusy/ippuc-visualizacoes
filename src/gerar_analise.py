
"""
gerar_analise.py
================
Lê as contribuições do Google Sheets (ou CSV local),
roda TF-IDF e LDA, e gera dois HTMLs standalone atualizados.

Dependências:
    pip install pandas scikit-learn gspread google-auth

Uso:
    # Lendo de CSV local (desenvolvimento):
    python gerar_analise.py --fonte csv --arquivo contribuicoes.csv

    # Lendo direto do Google Sheets (produção):
    python gerar_analise.py --fonte sheets \
        --sheet-id SEU_SHEET_ID \
        --aba Minuta \
        --credentials credentials.json

    # Com saída personalizada:
    python gerar_analise.py --fonte csv --arquivo contribuicoes.csv \
        --output-dir ./docs/assets

O script sempre sobrescreve palavras_chave.html e lda_vis.html na pasta de saída.
Ideal para rodar via cron, n8n (Execute Command) ou GitHub Actions.
"""

import argparse
import json
import re
import sys
import os
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────

N_TOPICOS = 5

NOMES_TOPICOS = [
    "Regulamentação Profissional",
    "Uso do Solo e Zoneamento",
    "Mobilidade e Infraestrutura",
    "Habitação e Inclusão Social",
    "Participação e Gestão",
]

CORES_TOPICOS = ["#2a9d8f", "#e76f51", "#e9c46a", "#457b9d", "#6a994e"]

STOPWORDS = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela
delas dele deles depois do dos e ela elas ele eles em entre era eram essa essas
esse esses esta estas este estes eu foi for foram há isso isto já lhe lhes
mais mas me mesmo meu meus minha minhas muito na nas não no nos nossa nossas
nosso nossos num numa o os ou para pela pelas pelo pelos per pode podem por
qual quando que quem se seu seus si sob sua suas também te teu teus toda todas
todo todos tu tua tuas um uma umas uns você vocês vos à às é ser ter tem têm
sendo tendo curitiba município municipal plano diretor art artigo lei federal
proposta deve deverá fica presente forma através pela pelo das dos sendo tendo
""".split())


# ── LEITURA DE DADOS ──────────────────────────────────────────────────────────

def ler_csv(caminho: str) -> pd.DataFrame:
    """Lê contribuições de um arquivo CSV local."""
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    print(f"[CSV] {len(df)} linhas lidas de {caminho}")
    return df


def ler_sheets(sheet_id: str, aba: str, credentials_path: str) -> pd.DataFrame:
    """
    Lê contribuições direto do Google Sheets via gspread.

    O arquivo credentials.json deve ser uma Service Account com acesso
    à planilha. Compartilhe a planilha com o email da service account.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("[ERRO] Instale: pip install gspread google-auth")
        sys.exit(1)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(sheet_id).worksheet(aba)
    dados = ws.get_all_records()
    df = pd.DataFrame(dados)
    print(f"[Sheets] {len(df)} linhas lidas da aba '{aba}'")
    return df


# ── PRÉ-PROCESSAMENTO ─────────────────────────────────────────────────────────

def limpar_texto(texto: str) -> str:
    """Remove ruído, stopwords e tokeniza para PT-BR."""
    texto = str(texto).lower()
    texto = re.sub(r"https?://\S+", "", texto)
    texto = re.sub(r"\\n", " ", texto)
    texto = re.sub(r"[^a-záàâãéèêíïóôõöúüçñ\s]", " ", texto)
    tokens = [t for t in texto.split() if len(t) > 3 and t not in STOPWORDS]
    return " ".join(tokens)


def preparar_corpus(df: pd.DataFrame):
    """
    Filtra linhas com conteúdo válido e cria coluna texto_limpo.
    Retorna DataFrame limpo.
    """
    df = df[df["conteudo"].notna()].copy()
    df = df[df["conteudo"].astype(str).str.strip() != ""].copy()
    df = df[df["conteudo"].astype(str).str.len() > 20].copy()
    df["texto_limpo"] = df["conteudo"].apply(limpar_texto)
    df = df[df["texto_limpo"].str.len() > 10].copy()
    df = df.reset_index(drop=True)
    print(f"[Corpus] {len(df)} documentos válidos após limpeza")
    return df


# ── ANÁLISE TF-IDF ────────────────────────────────────────────────────────────

def rodar_tfidf(df: pd.DataFrame) -> dict:
    """
    Calcula TF-IDF e retorna dicionário com:
    - palavras_top: ranking global dos termos
    - documentos: cada doc com submission_id, palavras-chave e texto
    """
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
    matriz = tfidf.fit_transform(df["texto_limpo"])
    features = tfidf.get_feature_names_out()

    # Score global
    scores = matriz.sum(axis=0).A1
    top_idx = scores.argsort()[::-1][:60]
    palavras_top = [
        {"palavra": features[i], "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]

    # Por documento
    documentos = []
    for idx, row in df.iterrows():
        vec = matriz[idx].toarray()[0]
        top_doc = vec.argsort()[::-1][:8]
        palavras_doc = [features[i] for i in top_doc if vec[i] > 0]
        documentos.append({
            "submission_id": str(row.get("submission_id", "")),
            "email": str(row.get("email", "")),
            "titulo": str(row.get("titulo", "")),
            "tipo": str(row.get("tipo", "")),
            "conteudo": str(row["conteudo"])[:300] + (
                "..." if len(str(row["conteudo"])) > 300 else ""
            ),
            "conteudo_completo": str(row["conteudo"]),
            "palavras_chave": palavras_doc,
        })

    return {"palavras_top": palavras_top, "documentos": documentos}


# ── ANÁLISE LDA ───────────────────────────────────────────────────────────────

def rodar_lda(df: pd.DataFrame) -> dict:
    """
    Treina LDA e retorna dicionário com:
    - topicos: lista com nome, tamanho e termos por tópico
    - docs: cada doc com tópico dominante e probabilidades
    - termos_globais: frequência global para as barras do pyLDAvis-style
    """
    cv = CountVectorizer(max_features=150, ngram_range=(1, 2), min_df=1)
    matriz = cv.fit_transform(df["texto_limpo"])
    features = cv.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICOS,
        random_state=42,
        max_iter=50,
        learning_method="batch",
    )
    lda_matrix = lda.fit_transform(matriz)

    # Frequências globais para o slider λ
    term_freq = matriz.sum(axis=0).A1
    total_tokens = term_freq.sum()
    p_w_global = term_freq / (total_tokens + 1e-10)

    # Dados por tópico
    df["_topico"] = lda_matrix.argmax(axis=1)
    topicos = []
    for i, comp in enumerate(lda.components_):
        p_w_dado_t = comp / comp.sum()
        relevancia = 0.6 * p_w_dado_t + 0.4 * (p_w_dado_t / (p_w_global + 1e-10))
        top_idx = relevancia.argsort()[::-1][:15]

        nome = NOMES_TOPICOS[i] if i < len(NOMES_TOPICOS) else f"Tópico {i}"
        topicos.append({
            "id": i,
            "nome": nome,
            "tamanho": int((df["_topico"] == i).sum()),
            "termos": [
                {
                    "termo": features[j],
                    "freq_topico": round(float(p_w_dado_t[j]), 5),
                    "freq_global": round(float(p_w_global[j]), 5),
                    "relevancia": round(float(relevancia[j]), 5),
                }
                for j in top_idx
            ],
        })

    # Dados por documento
    docs = []
    for idx, row in df.iterrows():
        probs = lda_matrix[idx].tolist()
        topico_dom = int(row["_topico"])
        nome_top = NOMES_TOPICOS[topico_dom] if topico_dom < len(NOMES_TOPICOS) else f"Tópico {topico_dom}"
        docs.append({
            "submission_id": str(row.get("submission_id", "")),
            "email": str(row.get("email", "")),
            "titulo": str(row.get("titulo", "")),
            "tipo": str(row.get("tipo", "")),
            "topico_dominante": topico_dom,
            "topico_nome": nome_top,
            "probabilidades": [round(p, 4) for p in probs],
            "conteudo": str(row["conteudo"])[:300] + (
                "..." if len(str(row["conteudo"])) > 300 else ""
            ),
            "conteudo_completo": str(row["conteudo"]),
        })

    # Termos globais (top 50 para barra de referência)
    termos_globais = [
        {"termo": features[j], "freq": int(term_freq[j])}
        for j in term_freq.argsort()[::-1][:50]
    ]

    return {"topicos": topicos, "docs": docs, "termos_globais": termos_globais}


# ── TEMPLATES HTML ────────────────────────────────────────────────────────────

HTML_PALAVRAS_CHAVE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Palavras-Chave — Minuta Plano Diretor Curitiba</title>
<style>
:root{--verde:#2a9d8f;--azul:#1f3a5c;--bg:#f8f7f4;--borda:#e0ddd8;--branco:#fff;--cinza:#6c757d}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:#1a1a2e}
header{background:var(--azul);color:white;padding:24px 32px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.3rem;font-weight:700}
header p{font-size:.85rem;color:#a8c4e0;margin-top:4px}
.badge{background:var(--verde);border-radius:20px;padding:4px 12px;font-size:.75rem;font-weight:600}
.container{max-width:1200px;margin:0 auto;padding:24px}
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}
.stat-box{background:var(--branco);border-radius:10px;padding:16px;text-align:center;border:.5px solid var(--borda)}
.stat-num{font-size:2rem;font-weight:800;color:var(--verde)}
.stat-label{font-size:.8rem;color:var(--cinza);margin-top:4px}
.grid-top{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
.card{background:var(--branco);border-radius:12px;padding:20px;border:.5px solid var(--borda)}
.card h2{font-size:1rem;font-weight:700;color:var(--azul);margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid var(--verde)}
#nuvem{display:flex;flex-wrap:wrap;gap:8px;align-items:center;min-height:180px}
.palavra-tag{cursor:pointer;border-radius:6px;padding:4px 10px;font-weight:600;transition:all .2s;background:#eaf7f5;color:var(--verde);border:1px solid transparent}
.palavra-tag:hover,.palavra-tag.ativa{background:var(--verde);color:white;transform:scale(1.05)}
.barra-item{margin-bottom:10px}
.barra-label{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:3px;color:#444}
.barra-bg{background:#eee;border-radius:4px;height:8px;overflow:hidden}
.barra-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--verde),var(--azul));transition:width .6s ease}
.filtros{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}
.filtros input,.filtros select{border:1.5px solid var(--borda);border-radius:8px;padding:8px 14px;font-size:.9rem;outline:none;background:white}
.filtros input:focus,.filtros select:focus{border-color:var(--verde)}
.filtros input{flex:1;min-width:200px}
#lista-contribuicoes{display:flex;flex-direction:column;gap:12px}
.contrib-card{background:var(--branco);border-radius:10px;padding:16px;border:.5px solid var(--borda);cursor:pointer;transition:all .2s}
.contrib-card:hover{border-color:var(--verde);box-shadow:0 2px 12px rgba(42,157,143,.15)}
.contrib-meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.tag{border-radius:12px;padding:2px 10px;font-size:.75rem;font-weight:600}
.tag-id{background:#f0f4ff;color:var(--azul);font-family:monospace;font-size:.7rem}
.tag-tipo{background:#fff8e1;color:#b07c00}
.tag-titulo{background:#eaf7f5;color:var(--verde);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.contrib-texto{font-size:.88rem;color:#333;line-height:1.55}
.contrib-palavras{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px}
.kw-pill{background:#f0f0f0;border-radius:10px;padding:2px 8px;font-size:.72rem;color:#555}
.kw-pill.match{background:var(--verde);color:white}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center;padding:20px}
.modal-overlay.open{display:flex}
.modal{background:white;border-radius:16px;max-width:700px;width:100%;max-height:80vh;overflow-y:auto;padding:28px;position:relative}
.modal-close{position:absolute;top:16px;right:16px;background:#f0f0f0;border:none;border-radius:50%;width:32px;height:32px;cursor:pointer;font-size:1.1rem}
.modal h3{font-size:1rem;color:var(--azul);margin-bottom:12px}
.modal-id{font-family:monospace;font-size:.78rem;color:var(--cinza);background:#f5f5f5;padding:6px 10px;border-radius:6px;margin-bottom:16px;word-break:break-all;line-height:1.6}
.modal-texto{font-size:.92rem;line-height:1.7;color:#222;white-space:pre-wrap}
.sem-resultado{text-align:center;padding:40px;color:var(--cinza);font-size:.95rem}
mark{background:#b2f0ea;border-radius:2px;padding:0 2px}
@media(max-width:768px){.grid-top{grid-template-columns:1fr}.stats-row{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<header>
  <div><h1>🔍 Análise de Palavras-Chave</h1><p>Minuta Interativa — Plano Diretor de Curitiba 2025</p></div>
  <div class="badge">TF-IDF</div>
</header>
<div class="container">
  <div class="stats-row">
    <div class="stat-box"><div class="stat-num" id="stat-docs">—</div><div class="stat-label">Contribuições analisadas</div></div>
    <div class="stat-box"><div class="stat-num" id="stat-palavras">—</div><div class="stat-label">Termos relevantes</div></div>
    <div class="stat-box"><div class="stat-num" id="stat-autores">—</div><div class="stat-label">Autores únicos</div></div>
  </div>
  <div class="grid-top">
    <div class="card"><h2>☁️ Nuvem de Palavras-Chave</h2><div id="nuvem">Carregando...</div><p style="font-size:.75rem;color:#999;margin-top:12px">Clique em uma palavra para filtrar</p></div>
    <div class="card"><h2>📊 Top 15 Termos — Score TF-IDF</h2><div id="barras"></div></div>
  </div>
  <div class="card">
    <h2>📄 Contribuições — Busca por texto, submission_id ou email</h2>
    <div class="filtros">
      <input type="text" id="busca-texto" placeholder="🔎 Buscar...">
      <select id="filtro-titulo"><option value="">Todos os títulos</option></select>
    </div>
    <div id="lista-contribuicoes"></div>
  </div>
</div>
<div class="modal-overlay" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="fecharModal()">✕</button>
    <h3 id="modal-titulo"></h3>
    <div class="modal-id" id="modal-id"></div>
    <div class="modal-texto" id="modal-texto"></div>
  </div>
</div>
<script>
const DADOS=__DADOS_PALAVRAS__;
let palavraAtiva=null;
function init(){
  document.getElementById('stat-docs').textContent=DADOS.documentos.length;
  document.getElementById('stat-palavras').textContent=DADOS.palavras_top.length;
  const autores=new Set(DADOS.documentos.map(d=>d.email).filter(Boolean));
  document.getElementById('stat-autores').textContent=autores.size;
  renderNuvem();renderBarras();renderTitulosFiltro();renderContribuicoes(DADOS.documentos);
  document.getElementById('busca-texto').addEventListener('input',filtrar);
  document.getElementById('filtro-titulo').addEventListener('change',filtrar);
}
function renderNuvem(){
  const max=DADOS.palavras_top[0].score;
  const el=document.getElementById('nuvem');el.innerHTML='';
  DADOS.palavras_top.slice(0,40).forEach(p=>{
    const ratio=p.score/max;const size=11+ratio*20;
    const tag=document.createElement('span');tag.className='palavra-tag';
    tag.textContent=p.palavra;tag.style.fontSize=size+'px';
    tag.title='Score TF-IDF: '+p.score;
    tag.onclick=()=>togglePalavra(p.palavra,tag);el.appendChild(tag);
  });
}
function renderBarras(){
  const max=DADOS.palavras_top[0].score;const el=document.getElementById('barras');el.innerHTML='';
  DADOS.palavras_top.slice(0,15).forEach(p=>{
    const pct=(p.score/max*100).toFixed(1);
    el.innerHTML+=`<div class="barra-item"><div class="barra-label"><span>${p.palavra}</span><span>${p.score.toFixed(3)}</span></div><div class="barra-bg"><div class="barra-fill" style="width:${pct}%"></div></div></div>`;
  });
}
function renderTitulosFiltro(){
  const sel=document.getElementById('filtro-titulo');
  [...new Set(DADOS.documentos.map(d=>d.titulo).filter(Boolean))].forEach(t=>{
    const opt=document.createElement('option');opt.value=t;opt.textContent=t.length>50?t.slice(0,50)+'...':t;sel.appendChild(opt);
  });
}
function togglePalavra(palavra,el){
  document.querySelectorAll('.palavra-tag').forEach(t=>t.classList.remove('ativa'));
  palavraAtiva=palavraAtiva===palavra?null:palavra;
  if(palavraAtiva)el.classList.add('ativa');
  filtrar();
}
function filtrar(){
  const busca=document.getElementById('busca-texto').value.toLowerCase().trim();
  const titulo=document.getElementById('filtro-titulo').value;
  let docs=DADOS.documentos;
  if(busca)docs=docs.filter(d=>d.conteudo_completo?.toLowerCase().includes(busca)||d.submission_id?.toLowerCase().includes(busca)||d.email?.toLowerCase().includes(busca));
  if(titulo)docs=docs.filter(d=>d.titulo===titulo);
  if(palavraAtiva)docs=docs.filter(d=>d.palavras_chave.includes(palavraAtiva)||d.conteudo_completo?.toLowerCase().includes(palavraAtiva));
  renderContribuicoes(docs,busca,palavraAtiva);
}
function renderContribuicoes(docs,busca='',palavra=''){
  const el=document.getElementById('lista-contribuicoes');
  if(!docs.length){el.innerHTML='<div class="sem-resultado">Nenhuma contribuição encontrada.</div>';return;}
  el.innerHTML=docs.map(d=>{
    let texto=d.conteudo||'';
    if(busca){const re=new RegExp(`(${busca.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`, 'gi');texto=texto.replace(re,'<mark>$1</mark>');}
    const pills=(d.palavras_chave||[]).map(p=>`<span class="kw-pill ${p===palavra?'match':''}">${p}</span>`).join('');
    return `<div class="contrib-card" onclick="abrirModal('${d.submission_id}')"><div class="contrib-meta"><span class="tag tag-id">📎 ${d.submission_id}</span><span class="tag tag-tipo">${d.tipo||'ideia'}</span><span class="tag tag-titulo" title="${d.titulo}">${(d.titulo||'').slice(0,50)}</span></div><div class="contrib-texto">${texto}</div><div class="contrib-palavras">${pills}</div></div>`;
  }).join('');
}
function abrirModal(sid){
  const doc=DADOS.documentos.find(d=>d.submission_id===sid);if(!doc)return;
  document.getElementById('modal-titulo').textContent=doc.titulo||'Contribuição';
  document.getElementById('modal-id').innerHTML=`<strong>submission_id:</strong> ${doc.submission_id}<br><strong>email:</strong> ${doc.email||'não informado'}<br><strong>tipo:</strong> ${doc.tipo}`;
  document.getElementById('modal-texto').textContent=doc.conteudo_completo||doc.conteudo;
  document.getElementById('modal').classList.add('open');
}
function fecharModal(){document.getElementById('modal').classList.remove('open');}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===document.getElementById('modal'))fecharModal();});
init();
</script>
</body></html>"""


HTML_LDA = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LDA — Minuta Plano Diretor Curitiba</title>
<style>
:root{--verde:#2a9d8f;--azul:#1f3a5c;--bg:#f8f7f4;--borda:#e0ddd8;--branco:#fff;--cinza:#6c757d}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:#1a1a2e}
header{background:var(--azul);color:white;padding:24px 32px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.3rem;font-weight:700}
header p{font-size:.85rem;color:#a8c4e0;margin-top:4px}
.badge{background:#e76f51;border-radius:20px;padding:4px 12px;font-size:.75rem;font-weight:600}
.container{max-width:1300px;margin:0 auto;padding:24px}
.layout{display:grid;grid-template-columns:340px 1fr;gap:20px}
.card{background:var(--branco);border-radius:12px;padding:20px;border:.5px solid var(--borda)}
.card h2{font-size:.95rem;font-weight:700;color:var(--azul);margin-bottom:14px;padding-bottom:10px;border-bottom:2px solid var(--verde)}
.topico-btn{width:100%;text-align:left;padding:12px 14px;border-radius:8px;border:1.5px solid var(--borda);background:white;cursor:pointer;margin-bottom:8px;transition:all .2s;display:flex;align-items:center;gap:10px}
.topico-btn:hover{border-color:var(--verde)}
.topico-btn.ativo{border-color:var(--verde);background:#eaf7f5}
.topico-dot{width:14px;height:14px;border-radius:50%;flex-shrink:0}
.topico-info{flex:1}
.topico-nome{font-size:.88rem;font-weight:600;color:#1a1a2e}
.topico-count{font-size:.75rem;color:var(--cinza);margin-top:2px}
.topico-badge{background:var(--verde);color:white;border-radius:10px;padding:2px 8px;font-size:.72rem;font-weight:700}
.lambda-box{margin:16px 0;padding:14px;background:#f5f3ee;border-radius:8px}
.lambda-box label{font-size:.82rem;color:#444;display:flex;justify-content:space-between;margin-bottom:8px}
.lambda-box input[type=range]{width:100%;accent-color:var(--verde)}
.lambda-desc{font-size:.75rem;color:var(--cinza);margin-top:6px;line-height:1.4}
.right-grid{display:flex;flex-direction:column;gap:16px}
#svg-bolhas{width:100%;height:200px}
.bolha{cursor:pointer;transition:opacity .2s}
.bolha:hover{opacity:.8}
.bolha.dimmed{opacity:.25}
.barra-lda{margin-bottom:9px;display:flex;align-items:center;gap:8px}
.barra-lda-label{width:160px;font-size:.78rem;color:#333;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:0;text-align:right}
.barra-lda-wrap{flex:1;position:relative;height:18px}
.barra-global{position:absolute;top:4px;height:10px;border-radius:3px;background:#ddd}
.barra-topico{position:absolute;top:4px;height:10px;border-radius:3px;opacity:.85}
.barra-lda-val{width:42px;font-size:.72rem;color:var(--cinza);text-align:right}
.legenda-barras{display:flex;gap:16px;font-size:.75rem;margin-bottom:12px}
.legenda-item{display:flex;align-items:center;gap:5px}
.legenda-cor{width:14px;height:10px;border-radius:2px}
#lista-topico{display:flex;flex-direction:column;gap:10px;max-height:420px;overflow-y:auto}
.contrib-mini{padding:12px 14px;border-radius:8px;border:.5px solid var(--borda);background:white;cursor:pointer;transition:all .2s}
.contrib-mini:hover{border-color:var(--verde)}
.contrib-mini-meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
.tag{border-radius:10px;padding:2px 8px;font-size:.72rem;font-weight:600}
.tag-id{background:#f0f4ff;color:var(--azul);font-family:monospace;font-size:.68rem}
.contrib-mini-texto{font-size:.83rem;color:#333;line-height:1.5}
.dist-bar{display:flex;height:32px;border-radius:8px;overflow:hidden;margin-bottom:8px}
.dist-seg{display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:700;color:white;transition:flex .5s ease;cursor:pointer}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center;padding:20px}
.modal-overlay.open{display:flex}
.modal{background:white;border-radius:16px;max-width:680px;width:100%;max-height:80vh;overflow-y:auto;padding:28px;position:relative}
.modal-close{position:absolute;top:16px;right:16px;background:#f0f0f0;border:none;border-radius:50%;width:32px;height:32px;cursor:pointer;font-size:1rem}
.modal-id{font-family:monospace;font-size:.78rem;color:var(--cinza);background:#f5f5f5;padding:8px 12px;border-radius:6px;margin-bottom:14px;line-height:1.6;word-break:break-all}
.modal-texto{font-size:.9rem;line-height:1.7;color:#222;white-space:pre-wrap}
.modal-probs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.prob-chip{border-radius:10px;padding:3px 10px;font-size:.74rem;font-weight:600}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div><h1>🧠 Análise de Tópicos — LDA</h1><p>Modelagem de tópicos latentes nas contribuições cidadãs</p></div>
  <div class="badge">LDA · __N_TOPICOS__ tópicos</div>
</header>
<div class="container">
  <div class="card" style="margin-bottom:20px">
    <h2>📊 Distribuição de Contribuições por Tópico</h2>
    <div class="dist-bar" id="dist-bar"></div>
    <div id="dist-legenda" style="display:flex;gap:12px;flex-wrap:wrap;font-size:.78rem"></div>
  </div>
  <div class="layout">
    <div style="display:flex;flex-direction:column;gap:16px">
      <div class="card"><h2>🗂️ Tópicos Identificados</h2><div id="topicos-lista"></div></div>
      <div class="card">
        <h2>⚙️ Parâmetro λ</h2>
        <div class="lambda-box">
          <label><span>Lambda: <strong id="lambda-val">0.6</strong></span><span style="color:var(--cinza)">0 → 1</span></label>
          <input type="range" id="lambda-slider" min="0" max="1" step="0.05" value="0.6">
          <p class="lambda-desc">λ=1: termos mais frequentes no tópico.<br>λ=0: termos mais exclusivos.<br>λ=0.6 é o equilíbrio recomendado.</p>
        </div>
        <h2 style="margin-top:8px">🫧 Tópicos por Volume</h2>
        <svg id="svg-bolhas" viewBox="0 0 320 180"></svg>
      </div>
    </div>
    <div class="right-grid">
      <div class="card">
        <h2 id="titulo-barras">Selecione um tópico para ver os termos</h2>
        <div class="legenda-barras">
          <div class="legenda-item"><div class="legenda-cor" style="background:#ddd"></div><span>Frequência global</span></div>
          <div class="legenda-item"><div class="legenda-cor" style="background:var(--verde)"></div><span>Relevância no tópico (λ)</span></div>
        </div>
        <div id="barras-lda"></div>
      </div>
      <div class="card">
        <h2 id="titulo-contrib">Contribuições do tópico selecionado</h2>
        <div id="lista-topico"><p style="color:var(--cinza);font-size:.88rem">Clique em um tópico.</p></div>
      </div>
    </div>
  </div>
</div>
<div class="modal-overlay" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="fecharModal()">✕</button>
    <h3 id="modal-titulo" style="font-size:1rem;color:var(--azul);margin-bottom:10px"></h3>
    <div class="modal-id" id="modal-id"></div>
    <div class="modal-probs" id="modal-probs"></div>
    <div class="modal-texto" id="modal-texto"></div>
  </div>
</div>
<script>
const DADOS=__DADOS_LDA__;
const CORES=__CORES__;
let topicoAtivo=0,lambda=0.6;
function init(){
  renderDistribuicao();renderTopicosBtns();renderBolhas();selecionarTopico(0);
  document.getElementById('lambda-slider').addEventListener('input',e=>{
    lambda=parseFloat(e.target.value);
    document.getElementById('lambda-val').textContent=lambda.toFixed(2);
    renderBarrasLDA(topicoAtivo);
  });
}
function renderDistribuicao(){
  const total=DADOS.docs.length;
  const bar=document.getElementById('dist-bar');const leg=document.getElementById('dist-legenda');
  bar.innerHTML='';leg.innerHTML='';
  DADOS.topicos.forEach((t,i)=>{
    const pct=total>0?(t.tamanho/total*100).toFixed(1):0;
    const seg=document.createElement('div');seg.className='dist-seg';
    seg.style.cssText=`flex:${t.tamanho||0.1};background:${CORES[i]};`;
    seg.textContent=pct>8?pct+'%':'';seg.title=t.nome+': '+t.tamanho+' docs';
    seg.onclick=()=>selecionarTopico(i);bar.appendChild(seg);
    leg.innerHTML+=`<div style="display:flex;align-items:center;gap:5px;cursor:pointer" onclick="selecionarTopico(${i})"><div style="width:12px;height:12px;border-radius:3px;background:${CORES[i]}"></div><span>${t.nome} (${t.tamanho})</span></div>`;
  });
}
function renderTopicosBtns(){
  const el=document.getElementById('topicos-lista');el.innerHTML='';
  DADOS.topicos.forEach((t,i)=>{
    el.innerHTML+=`<button class="topico-btn ${i===0?'ativo':''}" id="btn-topico-${i}" onclick="selecionarTopico(${i})"><div class="topico-dot" style="background:${CORES[i]}"></div><div class="topico-info"><div class="topico-nome">${t.nome}</div><div class="topico-count">Tópico ${i} · ${t.tamanho} contribuições</div></div><span class="topico-badge">${t.tamanho}</span></button>`;
  });
}
function renderBolhas(){
  const svg=document.getElementById('svg-bolhas');
  const total=DADOS.topicos.reduce((s,t)=>s+t.tamanho,0);
  const pos=[[60,90],[140,60],[220,90],[100,145],[180,145]];
  DADOS.topicos.forEach((t,i)=>{
    const[cx,cy]=pos[i]||[160,90];const r=20+(t.tamanho/(total||1))*55;
    svg.innerHTML+=`<circle class="bolha" cx="${cx}" cy="${cy}" r="${r}" fill="${CORES[i]}" fill-opacity=".8" onclick="selecionarTopico(${i})" id="bolha-${i}"><title>${t.nome}: ${t.tamanho}</title></circle><text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="middle" font-size="9" fill="white" font-weight="700" pointer-events="none" style="font-family:sans-serif">${t.tamanho}</text>`;
  });
}
function selecionarTopico(i){
  topicoAtivo=i;
  document.querySelectorAll('.topico-btn').forEach(b=>b.classList.remove('ativo'));
  document.getElementById('btn-topico-'+i)?.classList.add('ativo');
  document.querySelectorAll('.bolha').forEach((b,j)=>b.classList.toggle('dimmed',j!==i));
  renderBarrasLDA(i);renderContribuicoesTopico(i);
}
function calcRel(topico){
  return topico.termos.map(t=>{
    const rel=lambda*t.freq_topico+(1-lambda)*(t.freq_topico/(t.freq_global+1e-10));
    return{...t,rel_calc:rel};
  }).sort((a,b)=>b.rel_calc-a.rel_calc);
}
function renderBarrasLDA(i){
  const t=DADOS.topicos[i];
  document.getElementById('titulo-barras').textContent=`📌 Tópico ${i}: ${t.nome} — termos (λ=${lambda.toFixed(2)})`;
  const termos=calcRel(t);const maxRel=Math.max(...termos.map(x=>x.rel_calc));const maxG=Math.max(...termos.map(x=>x.freq_global));
  document.getElementById('barras-lda').innerHTML=termos.slice(0,15).map(term=>{
    const pR=(term.rel_calc/maxRel*100).toFixed(1);const pG=(term.freq_global/maxG*100).toFixed(1);
    return `<div class="barra-lda"><div class="barra-lda-label" title="${term.termo}">${term.termo}</div><div class="barra-lda-wrap"><div class="barra-global" style="width:${pG}%"></div><div class="barra-topico" style="width:${pR}%;background:${CORES[i]}"></div></div><div class="barra-lda-val">${(term.freq_topico*100).toFixed(2)}%</div></div>`;
  }).join('');
}
function renderContribuicoesTopico(i){
  const t=DADOS.topicos[i];
  document.getElementById('titulo-contrib').textContent=`📄 ${t.nome} — ${t.tamanho} contribuições`;
  const docs=DADOS.docs.filter(d=>d.topico_dominante===i);
  const el=document.getElementById('lista-topico');
  if(!docs.length){el.innerHTML='<p style="color:var(--cinza);font-size:.88rem">Nenhuma contribuição neste tópico.</p>';return;}
  el.innerHTML=docs.map(d=>{
    const prob=(d.probabilidades[i]*100).toFixed(1);
    return `<div class="contrib-mini" onclick="abrirModal('${d.submission_id}')"><div class="contrib-mini-meta"><span class="tag tag-id">📎 ${d.submission_id.slice(0,18)}…</span><span class="tag" style="background:${CORES[i]}22;color:${CORES[i]};font-size:.72rem;font-weight:600">${prob}% T${i}</span><span class="tag" style="background:#f0f4ff;color:var(--azul);font-size:.7rem">${(d.titulo||'').slice(0,35)}…</span></div><div class="contrib-mini-texto">${d.conteudo}</div></div>`;
  }).join('');
}
function abrirModal(sid){
  const doc=DADOS.docs.find(d=>d.submission_id===sid);if(!doc)return;
  document.getElementById('modal-titulo').textContent=doc.topico_nome;
  document.getElementById('modal-id').innerHTML=`<strong>submission_id:</strong> ${doc.submission_id}<br><strong>email:</strong> ${doc.email||'não informado'}<br><strong>título:</strong> ${doc.titulo}<br><strong>tipo:</strong> ${doc.tipo}`;
  document.getElementById('modal-probs').innerHTML=doc.probabilidades.map((p,i)=>`<span class="prob-chip" style="background:${CORES[i]}22;color:${CORES[i]};border:1px solid ${CORES[i]}44">T${i} ${(p*100).toFixed(1)}%</span>`).join('');
  document.getElementById('modal-texto').textContent=doc.conteudo_completo||doc.conteudo;
  document.getElementById('modal').classList.add('open');
}
function fecharModal(){document.getElementById('modal').classList.remove('open');}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===document.getElementById('modal'))fecharModal();});
init();
</script>
</body></html>"""


# ── GERAÇÃO DOS HTMLs ─────────────────────────────────────────────────────────

def gerar_html_palavras_chave(dados: dict, output_path: str):
    """Injeta os dados no template e salva o HTML."""
    html = HTML_PALAVRAS_CHAVE.replace(
        "__DADOS_PALAVRAS__",
        json.dumps(dados, ensure_ascii=False)
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {output_path}")


def gerar_html_lda(dados: dict, n_topicos: int, cores: list, output_path: str):
    """Injeta os dados no template LDA e salva o HTML."""
    html = HTML_LDA
    html = html.replace("__DADOS_LDA__", json.dumps(dados, ensure_ascii=False))
    html = html.replace("__N_TOPICOS__", str(n_topicos))
    html = html.replace("__CORES__", json.dumps(cores))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gera análises NLP da minuta do Plano Diretor")
    parser.add_argument("--fonte", choices=["csv", "sheets"], default="csv")
    parser.add_argument("--arquivo", default="contribuicoes.csv",
                        help="Caminho do CSV (quando --fonte=csv)")
    parser.add_argument("--sheet-id", default="",
                        help="ID do Google Sheets (quando --fonte=sheets)")
    parser.add_argument("--aba", default="Minuta",
                        help="Nome da aba no Sheets")
    parser.add_argument("--credentials", default="credentials.json",
                        help="Caminho do JSON da service account")
    parser.add_argument("--output-dir", default=".",
                        help="Pasta onde os HTMLs serão salvos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Leitura
    if args.fonte == "csv":
        df = ler_csv(args.arquivo)
    else:
        df = ler_sheets(args.sheet_id, args.aba, args.credentials)

    # 2. Pré-processamento
    df = preparar_corpus(df)

    if len(df) == 0:
        print("[AVISO] Nenhum documento válido encontrado. Abortando.")
        sys.exit(0)

    # 3. TF-IDF
    print("\n[TF-IDF] Calculando...")
    dados_pk = rodar_tfidf(df)

    # 4. LDA
    print("[LDA] Treinando modelo...")
    dados_lda = rodar_lda(df)

    # 5. Gerar HTMLs
    print("\n[HTML] Gerando arquivos...")
    gerar_html_palavras_chave(
        dados_pk,
        os.path.join(args.output_dir, "palavras_chave.html")
    )
    gerar_html_lda(
        dados_lda,
        N_TOPICOS,
        CORES_TOPICOS,
        os.path.join(args.output_dir, "lda_vis.html")
    )

    print(f"\n✅ Concluído! Arquivos salvos em: {args.output_dir}")
    print(f"   - palavras_chave.html")
    print(f"   - lda_vis.html")


if __name__ == "__main__":
    main()
