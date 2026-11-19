# Grupo 3 - K-Média dataset: Seeds
# Alunos: Felipe Echeverria Vilhalva, Alexandre Arruda, Lucas Hein e Sizenando
# Algoritmo de K-média clássico

import numpy as np
import pandas as pd

K = 3
np.random.seed(42)

# carrega os dados
colunas = ["area", "perimetro", "compacidade", "comp_grao",
           "larg_grao", "assimetria", "comp_sulco", "classe"]
df = pd.read_csv("data/seeds_dataset.txt", sep=r"\s+", header=None, names=colunas)
classes = df["classe"].to_numpy()                 # só para avaliar no final
X = df.drop(columns="classe").to_numpy(float)

# normaliza pra deixar os atributos na msm escala
X = (X - X.mean(axis=0)) / X.std(axis=0)


def distancia(a, b):
    """Distância euclidiana entre dois pontos."""
    return np.sqrt(np.sum((a - b) ** 2))

# escolhe os K pts aleatorios como centroids inciais
indices = np.random.choice(len(X), K, replace=False)
centroides = X[indices]

# repete: atribuicao e atualiza
grupos = np.zeros(len(X), dtype=int)
for iteracao in range(100):
    # atribui o ponto com base na distancia do centroide mais proximo
    novos_grupos = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        dists = [distancia(X[i], c) for c in centroides]
        novos_grupos[i] = np.argmin(dists)

    # so para quando os "grupos" nao mudam
    if iteracao > 0 and np.array_equal(novos_grupos, grupos):
        print(f"Convergiu na iteração {iteracao}")
        break
    grupos = novos_grupos

    # atuliza: centroide com a média dos pts do grupo
    for j in range(K):
        centroides[j] = X[grupos == j].mean(axis=0)

# resultado final no terminal
print("\nTamanho de cada grupo:", np.bincount(grupos))
print("\nGrupo x classe real (1=Kama, 2=Rosa, 3=Canadian):")
print(pd.crosstab(grupos, classes, rownames=["grupo"], colnames=["classe"]))

acertos = sum(np.bincount(classes[grupos == j]).max() for j in range(K))
print(f"\nPureza: {acertos / len(X):.1%}")


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