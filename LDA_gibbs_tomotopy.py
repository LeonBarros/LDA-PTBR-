"""
LDA em Python com amostragem de Gibbs (tomotopy)  -  equivalente ao topicmodels do R
Escolha de K por perplexidade em validacao cruzada.  Script completo e independente.

Instalar (uma vez), no terminal / Prompt de Comando:
    pip install tomotopy pypdf pandas matplotlib gensim
(o gensim aqui e usado so para a lista de stopwords em ingles)

Rodar:
    python LDA_gibbs_tomotopy.py
(uma janela abre para escolher a pasta com os PDFs)
Opcional: python LDA_gibbs_tomotopy.py "C:/caminho/da/pasta"
"""

import math
import random
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tomotopy as tp
from gensim.parsing.preprocessing import STOPWORDS
from pypdf import PdfReader

# --------------------------------------------------------------
# 0. CONFIGURACOES (edite aqui)
# --------------------------------------------------------------
K_MIN = 2            # menor numero de topicos a testar
K_MAX = 8            # maior numero de topicos a testar
K_FINAL = None       # None = usa o K de menor perplexidade; ou fixe, ex: 4
N_PALAVRAS = 35      # palavras exibidas por topico
SEED = 2024
N_FOLDS = 5          # dobras da validacao cruzada
ITER_CV = 300        # iteracoes de Gibbs na validacao cruzada (rapido)
ITER_FINAL = 1000    # iteracoes de Gibbs no modelo final

STOP_EXTRA = {
    "data", "not", "such", "they", "then", "was", "procedia", "can", "say",
    "one", "way", "use", "also", "will", "much", "need", "take", "even",
    "like", "particular", "rather", "said", "get", "well", "make", "ask",
    "come", "end", "first", "two", "help", "often", "may", "might", "see",
    "thing", "point", "post", "look", "right", "now", "think", "put", "set",
    "new", "good", "want", "sure", "kind", "large", "yes", "day", "etc",
    "since", "attempt", "lack", "seen", "little", "ever", "though", "found",
    "enough", "far", "away", "last", "never", "brief", "bit", "great", "lot",
    "fig", "let", "follow", "doi", "every",
}
STOPS = set(STOPWORDS) | STOP_EXTRA


# --------------------------------------------------------------
# 1. ESCOLHER A PASTA
# --------------------------------------------------------------
def escolher_pasta() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        pasta = filedialog.askdirectory(title="Selecione a pasta com os PDFs")
        root.destroy()
        if pasta:
            return Path(pasta)
    except Exception:
        pass
    return Path(input("Digite o caminho da pasta com os PDFs: ").strip().strip('"'))


# --------------------------------------------------------------
# 2. LER PDFs E LIMPAR O TEXTO
# --------------------------------------------------------------
def ler_pdf(caminho: Path) -> str:
    try:
        reader = PdfReader(str(caminho))
        return " ".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        print(f"  [aviso] falha ao ler {caminho.name}: {e}")
        return ""


def tokenizar(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"(ftp|https?)://\S+", " ", texto)       # URLs
    texto = re.sub(r"[^a-z\s]", " ", texto)                 # so letras
    return [t for t in texto.split() if len(t) >= 3 and t not in STOPS]


def filtrar_termos_raros(textos, no_below=2, no_above=0.95):
    """Equivale ao removeSparseTerms: mantem termos presentes em >= no_below docs
    e em no maximo no_above (fracao) dos docs."""
    n = len(textos)
    df = {}
    for t in textos:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    validos = {w for w, c in df.items() if c >= no_below and c / n <= no_above}
    return [[w for w in t if w in validos] for t in textos]


# --------------------------------------------------------------
# 3. FUNCOES DO MODELO
# --------------------------------------------------------------
def treinar(textos, k, iteracoes):
    modelo = tp.LDAModel(k=k, seed=SEED)
    for t in textos:
        modelo.add_doc(t)
    modelo.train(iteracoes, workers=1)
    return modelo


def perplexidade_teste(modelo, textos_teste):
    """exp(-loglik total / total de palavras) em documentos nao vistos."""
    docs = [modelo.make_doc(t) for t in textos_teste]
    _, ll = modelo.infer(docs, iterations=100, workers=1)
    n_palavras = sum(len(t) for t in textos_teste)
    return math.exp(-sum(ll) / n_palavras)


def main():
    matplotlib.use("Agg")  # salva graficos em arquivo, sem depender de janela

    pasta = escolher_pasta()
    pdfs = sorted(set(pasta.glob("*.pdf")) | set(pasta.glob("*.PDF")))
    print(f"PDFs encontrados: {len(pdfs)}")
    if len(pdfs) < 3:
        sys.exit("Sao necessarios pelo menos 3 PDFs.")

    nomes, textos = [], []
    for p in pdfs:
        toks = tokenizar(ler_pdf(p))
        if toks:
            nomes.append(p.name)
            textos.append(toks)
        else:
            print(f"  [aviso] {p.name}: sem texto extraivel (PDF escaneado?) - ignorado")

    # remove termos raros/comuns demais e docs que ficarem vazios
    textos = filtrar_termos_raros(textos)
    ok = [i for i, t in enumerate(textos) if len(t) > 0]
    nomes = [nomes[i] for i in ok]
    textos = [textos[i] for i in ok]
    vocab = {w for t in textos for w in t}
    print(f"Documentos no modelo: {len(textos)} | Termos: {len(vocab)}")

    # ----------------------------------------------------------
    # 4. ESCOLHER K (perplexidade em validacao cruzada)
    # ----------------------------------------------------------
    rng = random.Random(SEED)
    n = len(textos)
    n_folds = min(N_FOLDS, n)
    dobras = [i % n_folds for i in range(n)]
    rng.shuffle(dobras)

    print(f"\nCalculando perplexidade para K = {K_MIN} a {K_MAX} ...")
    ks, perps_medias = [], []
    for k in range(K_MIN, K_MAX + 1):
        perps = []
        for f in range(n_folds):
            treino = [t for t, d in zip(textos, dobras) if d != f]
            teste = [t for t, d in zip(textos, dobras) if d == f]
            try:
                m = treinar(treino, k, ITER_CV)
                perps.append(perplexidade_teste(m, teste))
            except Exception as e:
                print(f"  [aviso] K={k} dobra {f}: {e}")
        media = sum(perps) / len(perps) if perps else float("nan")
        ks.append(k)
        perps_medias.append(media)
        print(f"  K = {k} -> perplexidade media: {media:.1f}")

    validos = [(p, k) for k, p in zip(ks, perps_medias) if not math.isnan(p)]
    melhor_k = min(validos)[1]
    k_final = K_FINAL if K_FINAL is not None else melhor_k
    print(f"\nMelhor K (menor perplexidade): {melhor_k} | K usado: {k_final}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, perps_medias, marker="o")
    plt.axvline(k_final, color="red", linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Numero de topicos (K)")
    plt.ylabel("Perplexidade (validacao cruzada)")
    plt.title("Perplexidade por numero de topicos (menor = melhor)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pasta / "PYGibbs_perplexidade_por_K.png", dpi=150)
    plt.close()
    pd.DataFrame({"k": ks, "perplexidade": perps_medias}).to_csv(
        pasta / "PYGibbs_perplexidade_por_K.csv", index=False, sep=";",
        decimal=",", encoding="utf-8-sig")

    # ----------------------------------------------------------
    # 5. MODELO FINAL
    # ----------------------------------------------------------
    lda = treinar(textos, k_final, ITER_FINAL)

    # palavras por topico
    termos = {
        f"Topico_{t + 1}": [w for w, _ in lda.get_topic_words(t, top_n=N_PALAVRAS)]
        for t in range(k_final)
    }
    df_termos = pd.DataFrame(termos)

    # probabilidades por documento
    linhas = []
    for nome, doc in zip(nomes, lda.docs):
        dist = doc.get_topic_dist()
        linha = {"documento": nome}
        for t in range(k_final):
            linha[f"Topico_{t + 1}"] = float(dist[t])
        linhas.append(linha)
    df_probs = pd.DataFrame(linhas)

    # topico dominante
    cols = [f"Topico_{t + 1}" for t in range(k_final)]
    df_docs = pd.DataFrame({
        "documento": df_probs["documento"],
        "topico": df_probs[cols].values.argmax(axis=1) + 1,
    })

    print("\nPalavras principais por topico (top 10):")
    print(df_termos.head(10).to_string(index=False))
    print("\nTopico dominante por documento:")
    print(df_docs.to_string(index=False))

    # ----------------------------------------------------------
    # 6. GRAFICO: PALAVRAS PRINCIPAIS POR TOPICO
    # ----------------------------------------------------------
    ncols = min(3, k_final)
    nrows = -(-k_final // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    for t in range(nrows * ncols):
        ax = axes[t // ncols][t % ncols]
        if t < k_final:
            top = lda.get_topic_words(t, top_n=10)[::-1]
            ax.barh([w for w, _ in top], [p for _, p in top])
            ax.set_title(f"Topico {t + 1}")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(pasta / "PYGibbs_palavras_por_topico.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------
    # 7. SALVAR CSVs (ponto e virgula + UTF-8 com BOM: abre direto no Excel)
    # ----------------------------------------------------------
    opts = dict(index=False, sep=";", decimal=",", encoding="utf-8-sig")
    df_termos.to_csv(pasta / "PYGibbs_LDA_Terms_Topics.csv", **opts)
    df_docs.to_csv(pasta / "PYGibbs_LDA_DocsToTopics.csv", **opts)
    df_probs.to_csv(pasta / "PYGibbs_LDA_TopicProbabilities.csv", **opts)

    print(f"\nPRONTO! Arquivos salvos em: {pasta}")
    for f in sorted(pasta.glob("PYGibbs_*")):
        print("  ", f.name)


if __name__ == "__main__":   # obrigatorio no Windows
    main()
