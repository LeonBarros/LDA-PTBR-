"""
LDA em Python com gensim + CoherenceModel (c_v)  -  script completo e independente

Instalar (uma vez), no terminal / Prompt de Comando:
    pip install gensim pypdf pandas matplotlib

Rodar:
    python LDA_gensim.py
(uma janela abre para escolher a pasta com os PDFs)
Opcional: python LDA_gensim.py "C:/caminho/da/pasta"
"""

import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from pypdf import PdfReader

# --------------------------------------------------------------
# 0. CONFIGURACOES (edite aqui)
# --------------------------------------------------------------
K_MIN = 2            # menor numero de topicos a testar
K_MAX = 8            # maior numero de topicos a testar
K_FINAL = None       # None = usa o K de maior coerencia; ou fixe, ex: 4
N_PALAVRAS = 35      # palavras exibidas por topico
SEED = 2024
PASSES = 20          # passagens pelo corpus (mais = mais estavel)
ITERATIONS = 400

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


def main():
    matplotlib.use("Agg")  # salva graficos em arquivo, sem depender de janela

    pasta = escolher_pasta()
    pdfs = sorted(pasta.glob("*.pdf")) + sorted(pasta.glob("*.PDF"))
    pdfs = sorted(set(pdfs))
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

    # ----------------------------------------------------------
    # 3. DICIONARIO E CORPUS (equivale a matriz documento-termo)
    # ----------------------------------------------------------
    dicionario = Dictionary(textos)
    # no_below=2: termo precisa aparecer em >= 2 docs; no_above=0.95 ~ removeSparseTerms
    dicionario.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dicionario.doc2bow(t) for t in textos]

    # remove docs que ficaram vazios
    idx_ok = [i for i, bow in enumerate(corpus) if len(bow) > 0]
    nomes = [nomes[i] for i in idx_ok]
    textos = [textos[i] for i in idx_ok]
    corpus = [corpus[i] for i in idx_ok]
    print(f"Documentos no modelo: {len(corpus)} | Termos: {len(dicionario)}")

    # ----------------------------------------------------------
    # 4. ESCOLHER K PELA COERENCIA (c_v)
    # ----------------------------------------------------------
    print(f"\nCalculando coerencia c_v para K = {K_MIN} a {K_MAX} ...")
    ks, coerencias = [], []
    for k in range(K_MIN, K_MAX + 1):
        modelo = LdaModel(
            corpus=corpus, id2word=dicionario, num_topics=k,
            random_state=SEED, passes=PASSES, iterations=ITERATIONS,
            alpha="auto", eta="auto",
        )
        cm = CoherenceModel(
            model=modelo, texts=textos, dictionary=dicionario,
            coherence="c_v", processes=1,   # processes=1 evita problemas no Windows
        )
        c = cm.get_coherence()
        ks.append(k)
        coerencias.append(c)
        print(f"  K = {k} -> coerencia c_v: {c:.4f}")

    melhor_k = ks[max(range(len(ks)), key=lambda i: coerencias[i])]
    k_final = K_FINAL if K_FINAL is not None else melhor_k
    print(f"\nMelhor K (maior coerencia): {melhor_k} | K usado: {k_final}")

    # grafico de coerencia
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, coerencias, marker="o")
    plt.axvline(k_final, color="red", linestyle="--")
    plt.xticks(ks)
    plt.xlabel("Numero de topicos (K)")
    plt.ylabel("Coerencia c_v (maior = melhor)")
    plt.title("Coerencia por numero de topicos")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pasta / "PY_coerencia_por_K.png", dpi=150)
    plt.close()
    pd.DataFrame({"k": ks, "coerencia_c_v": coerencias}).to_csv(
        pasta / "PY_coerencia_por_K.csv", index=False, sep=";", decimal=",",
        encoding="utf-8-sig")

    # ----------------------------------------------------------
    # 5. MODELO FINAL
    # ----------------------------------------------------------
    lda = LdaModel(
        corpus=corpus, id2word=dicionario, num_topics=k_final,
        random_state=SEED, passes=PASSES, iterations=ITERATIONS,
        alpha="auto", eta="auto",
    )

    # palavras por topico
    termos = {}
    for t in range(k_final):
        palavras = [w for w, _ in lda.show_topic(t, topn=N_PALAVRAS)]
        termos[f"Topico_{t + 1}"] = palavras
    df_termos = pd.DataFrame(termos)

    # probabilidades por documento
    linhas = []
    for nome, bow in zip(nomes, corpus):
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        linha = {"documento": nome}
        for t in range(k_final):
            linha[f"Topico_{t + 1}"] = dist.get(t, 0.0)
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
            top = lda.show_topic(t, topn=10)[::-1]
            ax.barh([w for w, _ in top], [p for _, p in top])
            ax.set_title(f"Topico {t + 1}")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(pasta / "PY_palavras_por_topico.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------
    # 7. SALVAR CSVs (ponto e virgula + UTF-8 com BOM: abre direto no Excel)
    # ----------------------------------------------------------
    opts = dict(index=False, sep=";", decimal=",", encoding="utf-8-sig")
    df_termos.to_csv(pasta / "PY_LDA_Terms_Topics.csv", **opts)
    df_docs.to_csv(pasta / "PY_LDA_DocsToTopics.csv", **opts)
    df_probs.to_csv(pasta / "PY_LDA_TopicProbabilities.csv", **opts)

    print(f"\nPRONTO! Arquivos salvos em: {pasta}")
    for f in sorted(pasta.glob("PY_*")):
        print("  ", f.name)


if __name__ == "__main__":   # obrigatorio no Windows
    main()
