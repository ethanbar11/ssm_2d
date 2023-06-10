"""From original TF LMU implementation."""
import collections
import numpy as np
from tqdm import tqdm

# def mackey_glass(n_samples, l_seq=1000, tau=17, delta_t=10, seed=None, n_samples=1):
def mackey_glass(
    n_samples,
    l_seq=5000,
    l_predict=15,
    tau=17,
    washout=100,
    delta_t=10,
    center=True,
    seed=0,
):
    # Adapted from https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py
    '''
    mackey_glass(l_seq=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - l_seq: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    l_total = l_seq+l_predict+washout

    history_len = tau * delta_t
    # Initial conditions for the history of the system
    # timeseries = np.full(shape=(n_samples,), fill_value=1.2)

    if seed is not None:
        np.random.seed(seed)

    history = collections.deque(1.2 * np.ones((history_len, n_samples)) + 0.2 * \
            (np.random.rand(history_len, n_samples) - 0.5))
    inp = []

    for timestep in range(l_total):
        for _ in range(delta_t):
            xtau = history.popleft()
            timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                    0.1 * history[-1]) / delta_t
            inp.append(timeseries)
            history.append(timeseries)

    # Squash timeseries through tanh
    inp = np.stack(inp[::delta_t], axis=1) # (n_samples, l_total)
    # inp = np.tanh(inp - 1)
    X = inp[:, :, None]

    X = X[:, washout:, :]
    if center:
        X -= np.mean(X)  # global mean over all batches, approx -0.066
    Y = X[:, l_predict:, :]
    X = X[:, :-l_predict, :]
    # Y = X[:, :-l_predict, :]
    # X = X[:, l_predict:, :]
    assert X.shape == Y.shape
    return X, Y

# class MackeyGlassEvalDataset(torch.utils.data.TensorDataset):
#     def __init__(self, *args, **kwargs):
#         X, Y = mackey_glass(*args, **kwargs)
#         super().__init__(X, Y)



def cool_plot(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7,
                c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    (train_X, train_Y), (test_X, test_Y) = generate_data(128, 5000)
    cool_plot(train_X[0], train_Y[0])
