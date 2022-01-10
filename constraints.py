import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def main():
    X, y = make_moons(n_samples=20, random_state=100)
    const_mat = make_constraints(y, ratio=0.05, use_matrix=True)

    y_pred = KMeans(n_clusters=len(np.unique(y))).fit_predict(X)
    chk_mat = check_unsatisfied_constraints(const_mat, y_pred)
    C = np.count_nonzero(const_mat) // 2
    U = chk_mat.sum()

    fig = plt.figure(figsize=(6, 5))
    spec = fig.add_gridspec(ncols=1, nrows=1)

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(*X.T, c=y, s=40)

    for i, j in combinations(range(len(X)), 2):
        if const_mat[i, j] == 1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'g--', alpha=0.5)
        elif const_mat[i, j] == -1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'r--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("foo.png")
    plt.close()

if __name__ == '__main__':
    main()
