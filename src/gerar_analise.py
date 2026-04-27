"""
gerar_analise.py v2
===================
Gera todos os 6 HTMLs do dashboard IPPUC:
  index.html, palavras_chave.html, lda.html, pca.html, grafo.html, votos.html

Dependências:
    pip install pandas scikit-learn gspread google-auth networkx

Uso:
    # CSV local (desenvolvimento)
    python gerar_analise.py --fonte csv \
        --arquivo-minuta contribuicoes.csv \
        --arquivo-votos votos.csv

    # Google Sheets (producao)
    python gerar_analise.py --fonte sheets \
        --sheet-id SEU_SHEET_ID \
        --aba-minuta Contribuições \
        --aba-votos Votos \
        --credentials credentials.json \
        --output-dir docs/
"""

import argparse, json, re, sys, os, math
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────

N_TOPICOS = 5
NOMES_TOPICOS = [
    'Regulamentacao Profissional',
    'Uso do Solo e Zoneamento',
    'Mobilidade e Infraestrutura',
    'Habitacao e Inclusao Social',
    'Participacao e Gestao',
]
CORES_TOPICOS  = ['#00a859','#F07E31','#447EC0','#EA535E','#8E6CAD']
CORES_COMUNIDADES = ['#1a3a5c','#8E6CAD','#F07E31','#EA535E']
LETRAS_COMUNIDADES = ['A','B','C','D','E','F','G','H']

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
esta assim como bem ainda sobre caso apenas onde pois então desde seja cada
nova novo outras outros além deve desta deste nesta neste desta
""".split())

LOC_PATTERNS = [
    r'\b(avenida|av\.?|rua|r\.|travessa|tv\.?|alameda|al\.?|praça|pç\.?|rodovia|rod\.?|estrada)\s+\w[\w\s]{2,30}',
    r'\b(bairro|região|zona|setor|eixo|anel|corredor)\s+\w[\w\s]{1,20}',
    r'\b(manoel ribas|santa felicidade|cascatinha|centro cívico|marechal floriano|água verde|portão|rebouças|bigorrilho|cajuru|uberaba|pinheirinho|campo comprido|cidade industrial|boa vista|xaxim)\b',
    r'\b(zum-?\d|zr-?\d|ze-?\d|zs-?\d|az-?\d|vs-?\d)\b',
]
ORG_PATTERNS = [
    r'\b(ippuc|cft|crt|crea|confea|oab|cau|concitiba|reurb|cohab|urbs)\b',
    r'\b(conselho|câmara|secretaria|autarquia|fundação|instituto|departamento)\s+\w[\w\s]{1,25}',
    r'\b(lei federal|lei municipal|decreto|resolução|normativa)\s+n[oº°]?\s*[\d\.]+',
]
VERBOS = {
    'assegurar','garantir','promover','estabelecer','definir','regulamentar',
    'implementar','desenvolver','elaborar','criar','incluir','excluir',
    'ampliar','reduzir','otimizar','fiscalizar','monitorar','avaliar',
    'integrar','articular','fortalecer','incentivar','fomentar','viabilizar',
    'implantar','instalar','construir','reformar','recuperar','requalificar',
    'permitir','proibir','vedar','restringir','autorizar','licenciar',
    'participar','consultar','deliberar','aprovar','revisar','atualizar',
    'fica','ficam','deverá','deverão','será','serão','poderá','poderão',
    'constitui','constituem','visando','visa','tendo','sendo',
}
ADJETIVOS = {
    'social','urbano','urbana','sustentável','sustentáveis','integrado','integrada',
    'técnico','técnica','técnicos','técnicas','público','pública','públicos','públicas',
    'habitacional','ambiental','territorial','metropolitano','metropolitana',
    'econômico','econômica','cultural','histórico','histórica',
    'acessível','inclusivo','inclusiva','democrático','democrática',
    'regularizado','regularizada','qualificado','qualificada',
    'misto','mista','comercial','residencial','industrial',
    'estratégico','estratégica','prioritário','prioritária',
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def mascarar_email(email):
    if not email or '@' not in str(email):
        return 'nao informado'
    local, domain = str(email).split('@', 1)
    return local[0] + '****@' + domain

def limpar_display(texto):
    texto = str(texto)
    texto = texto.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
    texto = re.sub(r'[^\w\s\.,;:!?\-\(\)áàâãéèêíïóôõöúüçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇÑ]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def limpar_nlp(texto):
    texto = str(texto).lower()
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'\\n', ' ', texto)
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúüçñ\s]', ' ', texto)
    tokens = [t for t in texto.split() if len(t) > 3 and t not in STOPWORDS]
    return ' '.join(tokens)

def classificar_ner(texto):
    tl = texto.lower()
    res = {'loc': [], 'org': [], 'verb': [], 'adj': []}
    for pat in LOC_PATTERNS:
        for m in re.findall(pat, tl, re.IGNORECASE):
            t = m.strip() if isinstance(m, str) else ' '.join(m).strip()
            if t and len(t) > 3: res['loc'].append(t.lower())
    for pat in ORG_PATTERNS:
        for m in re.findall(pat, tl, re.IGNORECASE):
            t = m.strip() if isinstance(m, str) else ' '.join(m).strip()
            if t and len(t) > 2: res['org'].append(t.lower())
    for tok in re.findall(r'\b[\wÀ-ÿ]+\b', tl):
        if tok in VERBOS: res['verb'].append(tok)
        if tok in ADJETIVOS: res['adj'].append(tok)
    for k in res:
        res[k] = list(dict.fromkeys(res[k]))[:10]
    return res

def norm_dict(d):
    mn = min(d.values()); mx = max(d.values()); rng = mx - mn or 1
    return {k: round((v - mn) / rng, 4) for k, v in d.items()}

# ── LEITURA ───────────────────────────────────────────────────────────────────

def ler_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig')

def ler_sheets(sheet_id, aba, credentials_path):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print('[ERRO] pip install gspread google-auth')
        sys.exit(1)
    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly',
              'https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(sheet_id).worksheet(aba)
    dados = ws.get_all_records()
    return pd.DataFrame(dados)

# ── ANÁLISES ──────────────────────────────────────────────────────────────────

def processar_minuta(df_raw):
    total_bruto = len(df_raw)
    df = df_raw[df_raw['conteudo'].notna()].copy()
    df = df[df['conteudo'].astype(str).str.strip() != ''].copy()
    df = df[df['conteudo'].astype(str).str.len() > 20].copy()
    df['texto_limpo'] = df['conteudo'].apply(limpar_nlp)
    df = df[df['texto_limpo'].str.len() > 10].copy().reset_index(drop=True)
    df['conteudo_display'] = df['conteudo'].apply(limpar_display)
    df['email_mascarado'] = df.get('email', pd.Series([''] * len(df))).fillna('').apply(mascarar_email)
    print(f'[Minuta] bruto={total_bruto}, analisado={len(df)}')
    return df, total_bruto

def calcular_tfidf_ner(df, total_bruto):
    # NER
    df['ner'] = df['conteudo'].apply(classificar_ner)
    loc_c, org_c, verb_c, adj_c = Counter(), Counter(), Counter(), Counter()
    for _, row in df.iterrows():
        loc_c.update(row['ner']['loc']); org_c.update(row['ner']['org'])
        verb_c.update(row['ner']['verb']); adj_c.update(row['ner']['adj'])
    ner_global = {
        'loc':  [{'termo': k, 'freq': v} for k, v in loc_c.most_common(20)],
        'org':  [{'termo': k, 'freq': v} for k, v in org_c.most_common(20)],
        'verb': [{'termo': k, 'freq': v} for k, v in verb_c.most_common(20)],
        'adj':  [{'termo': k, 'freq': v} for k, v in adj_c.most_common(20)],
    }

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf.fit_transform(df['texto_limpo'])
    features = tfidf.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    top_idx = scores.argsort()[::-1]

    all_loc = set(t['termo'] for t in ner_global['loc'])
    all_org = set(t['termo'] for t in ner_global['org'])

    def cat_palavra(p):
        pl = p.lower()
        for loc in all_loc:
            if pl in loc or loc in pl: return 'loc'
        for org in all_org:
            if pl in org or org in pl: return 'org'
        if pl in VERBOS or any(pl.startswith(v[:4]) for v in VERBOS if len(v) > 4): return 'verb'
        if pl in ADJETIVOS: return 'adj'
        return 'other'

    vistas = set()
    palavras_top = []
    for i in top_idx:
        p = features[i]
        if p in vistas: continue
        partes = p.split()
        vistas.add(p)
        for parte in partes: vistas.add(parte)
        palavras_top.append({'palavra': p, 'score': round(float(scores[i]), 4), 'categoria': cat_palavra(p)})
        if len(palavras_top) >= 50: break

    docs_pk = []
    for idx, row in df.iterrows():
        vec = tfidf_matrix[idx].toarray()[0]
        top_doc = vec.argsort()[::-1][:8]
        pchave = list(dict.fromkeys([features[i] for i in top_doc if vec[i] > 0]))[:6]
        docs_pk.append({
            'submission_id': str(row.get('submission_id', '')),
            'email': row['email_mascarado'],
            'titulo': str(row.get('titulo', '')),
            'tipo': str(row.get('tipo', '')),
            'conteudo': row['conteudo_display'][:300] + ('...' if len(row['conteudo_display']) > 300 else ''),
            'conteudo_completo': row['conteudo_display'],
            'palavras_chave': pchave,
            'ner': row['ner'],
        })

    dados_pk = {
        'total_bruto': total_bruto,
        'total_analisado': len(df),
        'palavras_top': palavras_top,
        'ner_global': ner_global,
        'documentos': docs_pk,
    }
    print(f'[TF-IDF] {len(palavras_top)} termos')
    return dados_pk, tfidf_matrix, features, df

def calcular_lda(df, tfidf_matrix):
    cv = CountVectorizer(max_features=150, ngram_range=(1, 2), min_df=1)
    cv_matrix = cv.fit_transform(df['texto_limpo'])
    cv_features = cv.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=N_TOPICOS, random_state=42, max_iter=50)
    lda_matrix = lda.fit_transform(cv_matrix)
    term_freq = cv_matrix.sum(axis=0).A1
    p_w = term_freq / (term_freq.sum() + 1e-10)
    df['_topico'] = lda_matrix.argmax(axis=1)

    topicos = []
    for i, comp in enumerate(lda.components_):
        p_w_t = comp / comp.sum()
        rel = 0.6 * p_w_t + 0.4 * (p_w_t / (p_w + 1e-10))
        top_i = rel.argsort()[::-1][:15]
        topicos.append({
            'id': i, 'nome': NOMES_TOPICOS[i], 'tamanho': int((df['_topico'] == i).sum()),
            'termos': [{'termo': cv_features[j], 'freq_topico': round(float(p_w_t[j]), 5),
                        'freq_global': round(float(p_w[j]), 5), 'relevancia': round(float(rel[j]), 5)}
                       for j in top_i]
        })

    docs_lda = []
    for idx, row in df.iterrows():
        probs = lda_matrix[idx].tolist()
        td = int(row['_topico'])
        docs_lda.append({
            'submission_id': str(row.get('submission_id', '')),
            'email': row['email_mascarado'],
            'titulo': str(row.get('titulo', '')),
            'tipo': str(row.get('tipo', '')),
            'topico_dominante': td, 'topico_nome': NOMES_TOPICOS[td],
            'probabilidades': [round(p, 4) for p in probs],
            'conteudo': row['conteudo_display'][:300] + ('...' if len(row['conteudo_display']) > 300 else ''),
            'conteudo_completo': row['conteudo_display'],
        })

    dados_lda = {
        'topicos': topicos, 'docs': docs_lda,
        'termos_globais': [{'termo': cv_features[j], 'freq': int(term_freq[j])}
                           for j in term_freq.argsort()[::-1][:50]]
    }
    print(f'[LDA] {N_TOPICOS} topicos')
    return dados_lda, df

def calcular_pca(df, tfidf_matrix, features):
    X = tfidf_matrix.toarray()
    n_comp = min(X.shape[0] - 1, X.shape[1], 20)
    pca_full = PCA(n_components=n_comp, random_state=42)
    X_pca = pca_full.fit_transform(X)
    var_ratio = pca_full.explained_variance_ratio_
    var_cum = np.cumsum(var_ratio)
    n_80 = int(np.searchsorted(var_cum, 0.80)) + 1
    n_50 = int(np.searchsorted(var_cum, 0.50)) + 1
    scree = [{'pc': i+1, 'var': round(float(v)*100, 2), 'cum': round(float(c)*100, 2)}
             for i, v, c in zip(range(len(var_ratio)), var_ratio, var_cum)]
    loadings = pca_full.components_[:2]
    top_l1 = np.abs(loadings[0]).argsort()[::-1][:10]
    top_l2 = np.abs(loadings[1]).argsort()[::-1][:10]
    arrow_idx = list(dict.fromkeys(list(top_l1) + list(top_l2)))[:14]

    docs_pca = []
    for idx, row in df.iterrows():
        td = int(row['_topico'])
        docs_pca.append({
            'submission_id': str(row.get('submission_id', '')),
            'email': row['email_mascarado'],
            'titulo': str(row.get('titulo', '')),
            'tipo': str(row.get('tipo', '')),
            'topico': td, 'topico_nome': NOMES_TOPICOS[td],
            'x': round(float(X_pca[idx, 0]), 4),
            'y': round(float(X_pca[idx, 1]), 4),
            'conteudo': row['conteudo_display'][:200] + ('...' if len(row['conteudo_display']) > 200 else ''),
            'conteudo_completo': row['conteudo_display'],
        })

    arrows = [{'termo': features[i], 'x': round(float(loadings[0, i]), 4), 'y': round(float(loadings[1, i]), 4)}
              for i in arrow_idx]

    dados_pca = {
        'docs': docs_pca,
        'var_exp': [round(float(v)*100, 1) for v in var_ratio[:2]],
        'arrows': arrows, 'scree': scree,
        'n_80': n_80, 'n_50': n_50,
        'n_topicos': N_TOPICOS, 'nomes_topicos': NOMES_TOPICOS,
    }
    print(f'[PCA] var={[round(v*100,1) for v in var_ratio[:2]]}, n_80={n_80}')
    return dados_pca

def calcular_grafo(df, tfidf_matrix):
    THRESHOLD = 0.08
    sim_matrix = cosine_similarity(tfidf_matrix)
    n_docs = len(df)

    adj = {i: {} for i in range(n_docs)}
    arestas = []
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            sim = float(sim_matrix[i, j])
            if sim > THRESHOLD:
                adj[i][j] = sim; adj[j][i] = sim
                arestas.append({'source': i, 'target': j, 'weight': round(sim, 4)})

    grau      = {i: len(adj[i]) for i in range(n_docs)}
    grau_pond = {i: round(sum(adj[i].values()), 4) for i in range(n_docs)}

    # NetworkX para métricas avançadas
    G = nx.Graph()
    for i in range(n_docs): G.add_node(i)
    for a in arestas: G.add_edge(a['source'], a['target'], weight=a['weight'])

    densidade   = round(nx.density(G), 4)
    comunidades = nx.community.louvain_communities(G, seed=42)
    n_com       = len(comunidades)
    no_com      = {}
    for i, c in enumerate(comunidades):
        for nd in c: no_com[nd] = i

    # Excentricidade so no maior componente
    maior_comp = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(maior_comp)
    exc_main = nx.eccentricity(G_main)
    exc = {n: exc_main.get(n, -1) for n in G.nodes}
    pagerank   = nx.pagerank(G, weight='weight')
    eigenvec   = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    between    = nx.betweenness_centrality(G, weight='weight', normalized=True)
    clust      = nx.clustering(G, weight='weight')

    pr_n  = norm_dict(pagerank)
    ev_n  = norm_dict(eigenvec)
    bt_n  = norm_dict(between)
    cl_n  = norm_dict(clust)
    ex_n  = norm_dict(exc)

    def bridging_coef(node):
        vizinhos = list(adj[node].keys())
        if not vizinhos: return 0.0
        g = grau[node]
        if g == 0: return 0.0
        num = (1/g)**2
        denom = sum((1/(grau[v]+1e-10))**2 for v in vizinhos)
        return round(num/denom, 4) if denom else 0.0

    bc_nos = {i: bridging_coef(i) for i in range(n_docs)}

    for a in arestas:
        a['bridging'] = round((bc_nos[a['source']] + bc_nos[a['target']]) / 2, 4)

    nos_conn = set()
    for a in arestas: nos_conn.add(a['source']); nos_conn.add(a['target'])

    nos_raw = []
    for idx, row in df.iterrows():
        if idx not in nos_conn: continue
        td = int(row['_topico'])
        nos_raw.append({
            'id': idx,
            'submission_id': str(row.get('submission_id', '')),
            'email': row['email_mascarado'],
            'titulo': str(row.get('titulo', ''))[:60],
            'topico': td, 'topico_nome': NOMES_TOPICOS[td],
            'tipo': str(row.get('tipo', '')),
            'conteudo': row['conteudo_display'][:200] + ('...' if len(row['conteudo_display']) > 200 else ''),
            'conteudo_completo': row['conteudo_display'],
            'grau': grau[idx], 'grau_pond': grau_pond[idx],
            'bridging': bc_nos[idx],
            'comunidade': no_com.get(idx, 0),
            'excentricidade': exc.get(idx, -1),
            'pagerank': round(pagerank.get(idx, 0), 5),
            'eigenvector': round(eigenvec.get(idx, 0), 5),
            'betweenness': round(between.get(idx, 0), 5),
            'clustering': round(clust.get(idx, 0), 5),
            'pr_n': pr_n.get(idx, 0), 'ev_n': ev_n.get(idx, 0),
            'bt_n': bt_n.get(idx, 0), 'cl_n': cl_n.get(idx, 0),
            'ex_n': ex_n.get(idx, 0),
        })

    id_map = {n['id']: i for i, n in enumerate(nos_raw)}
    for n in nos_raw: n['id'] = id_map[n['id']]
    arestas_f = [{'source': id_map[a['source']], 'target': id_map[a['target']],
                  'weight': a['weight'], 'bridging': a['bridging']}
                 for a in arestas if a['source'] in id_map and a['target'] in id_map]

    gmax  = max((n['grau'] for n in nos_raw), default=1)
    gpmax = max((n['grau_pond'] for n in nos_raw), default=1.0)

    dados_grafo = {
        'nos': nos_raw, 'arestas': arestas_f,
        'nomes_topicos': NOMES_TOPICOS,
        'grau_max': gmax, 'grau_pond_max': gpmax,
        'metricas_globais': {
            'densidade': densidade, 'n_comunidades': n_com,
            'n_nos': len(nos_raw), 'n_arestas': len(arestas_f),
        }
    }
    print(f'[Grafo] {len(nos_raw)} nos, {len(arestas_f)} arestas, {n_com} comunidades')
    return dados_grafo

def calcular_votos(df_votos):
    if df_votos is None or df_votos.empty:
        return None
    df_votos['voto']  = df_votos['voto'].astype(str).str.strip()
    df_votos['nivel'] = df_votos['nivel'].astype(str).str.strip().fillna('titulo')
    df_votos['nome']  = df_votos['nome'].astype(str).str.strip()

    votantes   = df_votos['submission_id'].nunique()
    total      = len(df_votos)
    adequado   = int((df_votos['voto'] == 'adequado').sum())
    precisa    = int((df_votos['voto'] == 'precisa_ajuste').sum())

    dados_por_nivel = {}
    for nivel in ['titulo', 'capitulo', 'secao', 'subsecao']:
        sub = df_votos[df_votos['nivel'] == nivel]
        if sub.empty:
            dados_por_nivel[nivel] = []
            continue
        grp = sub.groupby('nome')['voto'].value_counts().unstack(fill_value=0).reset_index()
        if 'adequado'      not in grp.columns: grp['adequado'] = 0
        if 'precisa_ajuste' not in grp.columns: grp['precisa_ajuste'] = 0
        grp['total'] = grp['adequado'] + grp['precisa_ajuste']
        grp = grp.sort_values('total', ascending=False)
        dados_por_nivel[nivel] = [
            {'nome': r['nome'], 'adequado': int(r['adequado']), 'precisa': int(r['precisa_ajuste'])}
            for _, r in grp.iterrows()
        ]

    print(f'[Votos] votantes={votantes}, total={total}, adequado={adequado}, precisa={precisa}')
    return {
        'kpis': {'votantes': votantes, 'total': total, 'adequado': adequado, 'precisa': precisa},
        'niveis': dados_por_nivel,
    }

# ── GERAÇÃO DOS HTMLs ─────────────────────────────────────────────────────────

def salvar_htmls(dados_pk, dados_lda, dados_pca, dados_grafo, dados_votos, output_dir):
    """Atualiza os dados JS embutidos em cada HTML sem alterar o layout."""
    import re as _re

    def atualizar_var(nome_arquivo, var_name, novo_json):
        caminho = os.path.join(output_dir, nome_arquivo)
        if not os.path.exists(caminho):
            print(f'[AVISO] {nome_arquivo} nao encontrado em {output_dir} — pulando')
            return
        with open(caminho, encoding='utf-8') as f:
            html = f.read()
        padrao = f'const {var_name}='
        inicio = html.find(padrao)
        if inicio == -1:
            print(f'[AVISO] {var_name} nao encontrado em {nome_arquivo}')
            return
        pos = inicio + len(padrao)
        depth = 0; in_str = False; escape = False; end = pos
        while end < len(html):
            c = html[end]
            if escape: escape = False
            elif c == '\\' and in_str: escape = True
            elif c == '"' and not escape: in_str = not in_str
            elif not in_str:
                if c in '{[': depth += 1
                elif c in '}]':
                    depth -= 1
                    if depth == 0: end += 1; break
            end += 1
        novo_html = html[:inicio] + f'const {var_name}={novo_json}' + html[end:]
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(novo_html)
        print(f'[OK] {nome_arquivo} — {var_name} atualizado')

    atualizar_var('palavras_chave.html', 'D',    json.dumps(dados_pk,    ensure_ascii=False))
    atualizar_var('lda.html',            'D',    json.dumps(dados_lda,   ensure_ascii=False))
    atualizar_var('pca.html',            'D',    json.dumps(dados_pca,   ensure_ascii=False))
    atualizar_var('grafo.html',          'DALL', json.dumps(dados_grafo, ensure_ascii=False))
    if dados_votos:
        atualizar_var('votos.html', 'D', json.dumps(dados_votos, ensure_ascii=False))

    # Atualizar KPIs do index.html
    index_path = os.path.join(output_dir, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            html = f.read()
        kpis = dados_votos['kpis'] if dados_votos else {'votantes':0,'total':1,'adequado':0,'precisa':0}
        tb = dados_pk['total_bruto']; ta = dados_pk['total_analisado']
        ad = kpis['adequado']; pr = kpis['precisa']
        vv = kpis['total'] or 1; vt = kpis['votantes']
        pct_ad = round(ad/vv*100,1); pct_pr = round(pr/vv*100,1)
        replacements = [
            (r'(<div class="knum" style="color:var\(--az\);">)\d+(</div>\s*<div class="klbl">Votantes)', f'\\g<1>{vt}\\g<2>'),
            (r'(<div class="knum" style="color:var\(--azd\);">)\d+(</div>\s*<div class="klbl">Votos)', f'\\g<1>{vv}\\g<2>'),
            (r'(<div class="knum" style="color:var\(--la\);">)\d+(</div>\s*<div class="klbl">Precisa)', f'\\g<1>{pr}\\g<2>'),
            (r'(<div class="knum" style="color:var\(--v\);">)\d+(</div>\s*<div class="klbl">Adequado)', f'\\g<1>{ad}\\g<2>'),
            (r'(<div class="knum" style="color:var\(--azd\);">)\d+(</div>\s*<div class="klbl">Contribuicoes recebidas)', f'\\g<1>{tb}\\g<2>'),
            (r'(<div class="knum" style="color:var\(--v\);">)\d+(</div>\s*<div class="klbl">Contribuicoes analisadas)', f'\\g<1>{ta}\\g<2>'),
            (r'(class="vbar-ad" style="width:)[^"]+(")', f'\\g<1>{pct_ad}%\\g<2>'),
            (r'(class="vbar-pr" style="width:)[^"]+(")', f'\\g<1>{pct_pr}%\\g<2>'),
            (r'(Adequado )\d+\.?\d*(%)', f'\\g<1>{pct_ad}\\g<2>'),
            (r'(Precisa de Ajuste )\d+\.?\d*(%)', f'\\g<1>{pct_pr}\\g<2>'),
        ]
        for pat, rep in replacements:
            html = _re.sub(pat, rep, html)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print('[OK] index.html KPIs atualizados')


def main():
    parser = argparse.ArgumentParser(description='Gera dashboard NLP IPPUC — Plano Diretor Curitiba')
    parser.add_argument('--fonte', choices=['csv','sheets'], default='csv')
    parser.add_argument('--arquivo-minuta', default='contribuicoes.csv')
    parser.add_argument('--arquivo-votos',  default='votos.csv')
    parser.add_argument('--sheet-id',       default='')
    parser.add_argument('--aba-minuta',     default='Minuta')
    parser.add_argument('--aba-votos',      default='Votos')
    parser.add_argument('--credentials',    default='credentials.json')
    parser.add_argument('--output-dir',     default='docs/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Ler dados
    if args.fonte == 'csv':
        df_raw    = ler_csv(args.arquivo_minuta)
        try:    df_votos = ler_csv(args.arquivo_votos)
        except: df_votos = None; print('[AVISO] arquivo de votos nao encontrado')
    else:
        df_raw   = ler_sheets(args.sheet_id, args.aba_minuta, args.credentials)
        try:    df_votos = ler_sheets(args.sheet_id, args.aba_votos, args.credentials)
        except: df_votos = None; print('[AVISO] aba de votos nao encontrada')

    # 2. Processar
    df, total_bruto = processar_minuta(df_raw)
    if len(df) == 0:
        print('[ERRO] Nenhum documento valido. Abortando.')
        sys.exit(0)

    dados_pk, tfidf_matrix, features, df = calcular_tfidf_ner(df, total_bruto)
    dados_lda, df = calcular_lda(df, tfidf_matrix)
    dados_pca     = calcular_pca(df, tfidf_matrix, features)
    dados_grafo   = calcular_grafo(df, tfidf_matrix)
    dados_votos   = calcular_votos(df_votos)

    # 3. Salvar
    salvar_htmls(dados_pk, dados_lda, dados_pca, dados_grafo, dados_votos, args.output_dir)
    print('\nConcluido!')

if __name__ == '__main__':
    main()
