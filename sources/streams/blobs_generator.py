import numpy as np


def make_normal_blobs(centers, n_samples=1000, weights=None, random_state=None):
    rng = np.random.default_rng(random_state)
    n_classes = len(centers)

    if weights is None:
        weights = [1 / n_classes] * n_classes

    weights = np.array(weights)
    assert weights.sum() == 1

    class_samples = (weights * n_samples).astype(int)
    assert class_samples.sum() == n_samples


    X_, y_ = [], []
    for l, (c, n) in enumerate(zip(centers, class_samples)):
        std = np.eye(len(c))
        X_.append(rng.multivariate_normal(c, std, size=n))
        y_.append([l] * n)

    shuffle_ind = np.arange(n_samples)
    rng.shuffle(shuffle_ind)
    return np.concatenate(X_)[shuffle_ind], np.concatenate(y_)[shuffle_ind]


def make_beta_blobs(centers, radius=None, a=1, b=1, n_samples=1000, weights=None, random_state=None):
    rng = np.random.default_rng(random_state)
    n_classes = len(centers)
    n_dims = len(centers[0])

    if radius is None:
        radius = np.ones(n_classes)

    if weights is None:
        weights = [1 / n_classes] * n_classes

    weights = np.array(weights)
    # assert weights.sum() == 1

    class_samples = (weights * n_samples).astype(int)
    if class_samples.sum() != n_samples:
        min_class = np.argmin(class_samples)
        class_samples[min_class] += 1

    assert class_samples.sum() == n_samples


    X_, y_ = [], []
    for l, (c, r, n) in enumerate(zip(centers, radius, class_samples)):
        rv = rng.normal(size=(n, n_dims))
        norm = np.linalg.norm(rv, axis=1)
        rv = rv / norm[:, np.newaxis]

        dist = np.sqrt(rng.beta(a, b, size=n)) * r
        rv = rv * dist[:, np.newaxis]

        rv = rv + c

        X_.append(rv)
        y_.append([l] * n)

    shuffle_ind = np.arange(n_samples)
    rng.shuffle(shuffle_ind)
    return np.concatenate(X_)[shuffle_ind], np.concatenate(y_)[shuffle_ind]


def main():
    import matplotlib.pyplot as plt
    X, y = make_beta_blobs([[2.0, 2.0], [-5, -5]], radius=[2, 2], n_samples=1000, weights=[0.1, 0.9])
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(*X.T, c=y, s=10, alpha=0.8)
    plt.tight_layout()
    plt.savefig("foo.png")
    plt.close()

if __name__ == '__main__':
    main()
