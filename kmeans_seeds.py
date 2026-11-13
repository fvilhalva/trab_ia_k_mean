"""K-médias (K-means) implementado do zero no dataset Seeds (UCI #236).

Usa apenas numpy, pandas e matplotlib (para o gráfico). Sem scikit-learn,
PyTorch ou TensorFlow.

Uso:  python kmeans_seeds.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "seeds_dataset.txt"
COLUNAS = ["area", "perimetro", "compacidade", "comp_grao",
           "larg_grao", "assimetria", "comp_sulco", "classe"]


def carregar():
    df = pd.read_csv(DATA, sep=r"\s+", header=None, names=COLUNAS)
    return df.drop(columns="classe").to_numpy(float), df["classe"].to_numpy(int)


def normalizar(X):
    """Z-score: evita que atributos de escala maior dominem a distância."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def distancias(X, C):
    """Distância euclidiana ao quadrado entre cada ponto e cada centróide."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans_pp(X, k, rng):
    """Inicialização k-means++ (espalha os centróides iniciais)."""
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = distancias(X, np.array(C)).min(axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


def kmeans(X, k, rng, max_iter=300, tol=1e-9):
    C = kmeans_pp(X, k, rng)
    for _ in range(max_iter):
        rot = distancias(X, C).argmin(axis=1)              # passo de atribuição
        novo = np.array([X[rot == j].mean(axis=0) if (rot == j).any()
                         else X[rng.integers(len(X))] for j in range(k)])  # atualização
        mov = np.abs(novo - C).max()
        C = novo
        if mov < tol:
            break
    d = distancias(X, C)
    return C, d.argmin(axis=1), d.min(axis=1).sum()        # inércia (SSE)


def melhor_de_n(X, k, n_init=20, seed=0):
    rng = np.random.default_rng(seed)
    return min((kmeans(X, k, rng) for _ in range(n_init)), key=lambda r: r[2])


def silhueta(X, rot):
    """Coeficiente de silhueta médio (implementação própria)."""
    D = np.sqrt(distancias(X, X))
    ks = np.unique(rot)
    s = np.zeros(len(X))
    for i in range(len(X)):
        mesmo = rot == rot[i]
        if mesmo.sum() == 1:
            continue
        a = D[i, mesmo].sum() / (mesmo.sum() - 1)
        b = min(D[i, rot == c].mean() for c in ks if c != rot[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def acuracia_cluster(rot, classe):
    """Acurácia comparando cada cluster com a classe majoritária (pureza)."""
    return sum(np.bincount(classe[rot == c]).max() for c in np.unique(rot)) / len(rot)


def main():
    X, y = carregar()
    Xn = normalizar(X)

    print("== Método do cotovelo + silhueta ==")
    print(f"{'k':>2} {'inércia':>10} {'silhueta':>9}")
    for k in range(2, 9):
        _, rot, sse = melhor_de_n(Xn, k)
        print(f"{k:>2} {sse:>10.2f} {silhueta(Xn, rot):>9.3f}")

    print("\n== k=3 (o dataset tem 3 variedades: Kama, Rosa, Canadian) ==")
    C, rot, sse = melhor_de_n(Xn, 3)
    print(f"Inércia: {sse:.2f} | silhueta: {silhueta(Xn, rot):.3f}")
    print(f"Pureza vs. classes reais: {acuracia_cluster(rot, y):.3f}")
    print("Tabela cluster x classe real:")
    print(pd.crosstab(rot, y, rownames=["cluster"], colnames=["classe"]))
    print("\nCentróides (escala original):")
    Corig = C * X.std(axis=0) + X.mean(axis=0)
    print(pd.DataFrame(Corig, columns=COLUNAS[:-1]).round(3))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # PCA simples via SVD só para visualização em 2D
        U, S, Vt = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)
        P, Pc = Xn @ Vt[:2].T, C @ Vt[:2].T
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a, lab, t in zip(ax, (rot, y), ("Clusters K-médias", "Classes reais")):
            a.scatter(P[:, 0], P[:, 1], c=lab, cmap="viridis", s=18)
            a.set_title(t)
        ax[0].scatter(Pc[:, 0], Pc[:, 1], c="red", marker="X", s=150, label="centróides")
        ax[0].legend()
        out = Path(__file__).parent / "kmeans_seeds.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nGráfico salvo em {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
