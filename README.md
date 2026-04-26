[Uploading README.md…]()

# 📊 Análise NLP — Minuta Plano Diretor Curitiba

Pipeline automático que gera visualizações interativas das contribuições cidadãs
coletadas pela [Minuta Interativa](https://minuta-planodiretor.vercel.app).

---

## Estrutura

```
.
├── .github/
│   └── workflows/
│       └── atualizar_analise.yml   ← disparado pelo n8n ou cron
├── src/
│   └── gerar_analise.py            ← script NLP (TF-IDF + LDA)
├── docs/                           ← GitHub Pages (publicado automaticamente)
│   ├── index.html
│   ├── palavras_chave.html
│   └── lda_vis.html
└── README.md
```

---

## Setup — Passo a passo

### 1. Secrets do GitHub

Vá em **Settings → Secrets and variables → Actions** e crie:

| Secret | Valor |
|--------|-------|
| `GOOGLE_CREDENTIALS_JSON` | Conteúdo completo do `credentials.json` da service account |
| `SHEET_ID` | `1X0bJEWsMeVEykiuXCSKjSeWpvmVeQTyOVJBd0zeJT1k` |

### 2. Service Account

A service account já existe no projeto:
```
scraper-planilha@plasma-light-283517.iam.gserviceaccount.com
```

Compartilhe a planilha `Minuta` com esse email (permissão de **leitura**).

### 3. GitHub Pages

Vá em **Settings → Pages**:
- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

A URL pública será:
```
https://cassiusbusy.github.io/minuta-planodiretor/
```

### 4. Trigger no n8n

Adicione um node **HTTP Request** após o `Append row in sheet_minuta`:

```
Método: POST
URL: https://api.github.com/repos/cassiusbusy/minuta-planodiretor/dispatches
Headers:
  Authorization: Bearer SEU_GITHUB_TOKEN
  Accept: application/vnd.github+json
  X-GitHub-Api-Version: 2022-11-28
Body (JSON):
  { "event_type": "atualizar-analise" }
```

Para gerar o token: GitHub → **Settings → Developer settings →
Personal access tokens → Fine-grained tokens** → permissão `Actions: write`
no repositório `minuta-planodiretor`.

---

## Uso local (desenvolvimento)

```bash
pip install pandas scikit-learn gspread google-auth

# Com CSV local
python src/gerar_analise.py \
  --fonte csv \
  --arquivo contribuicoes.csv \
  --output-dir docs/

# Com Google Sheets
python src/gerar_analise.py \
  --fonte sheets \
  --sheet-id 1X0bJEWsMeVEykiuXCSKjSeWpvmVeQTyOVJBd0zeJT1k \
  --aba Minuta \
  --credentials credentials.json \
  --output-dir docs/
```

---

## Como funciona o pipeline

```
Usuário submete contribuição
        ↓
   Lovable/Vercel
        ↓
  Edge Function Supabase (submit-proposal)
        ↓
  Webhook n8n
        ↓
  Prepara Dados → Drive + Email + Sheets
        ↓
  HTTP Request → GitHub API (repository_dispatch)
        ↓
  GitHub Actions (atualizar_analise.yml)
        ↓
  gerar_analise.py lê Google Sheets
        ↓
  Gera palavras_chave.html + lda_vis.html
        ↓
  Commit automático → GitHub Pages
        ↓
  URL pública atualizada em ~2 min
```
