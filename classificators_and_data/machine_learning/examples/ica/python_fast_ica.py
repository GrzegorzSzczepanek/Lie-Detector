import numpy as np


class ICA:
    def __init__(self):
        pass

    def center(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        centered = x - mean
        return centered, mean

    def covariance(self, x):
        # mean_x = np.mean(x, axis=-1, keepdims=True)
        
        m = self.center(x)[0]
        
        # calculate the covariance matrix for each sample
        cov_matrices = []
        for i in range(m.shape[-1]):
            cov_matrix = m[..., i].T.dot(m[..., i]) / (m.shape[-1] - 1)
            cov_matrices.append(cov_matrix)
        
        total_cov_matrix = np.sum(np.stack(cov_matrices), axis=0) / len(cov_matrices)
        
        return total_cov_matrix

    def whiten(self, x):
        coVarM = self.covariance(x)
        U, S, V = np.linalg.svd(coVarM)
        d = np.diag(1.0 / np.sqrt(S))
        whiteM = np.dot(U, np.dot(d, U.T))
        Xw = np.dot(whiteM, x)
        return Xw, whiteM

    def fastICA(self, signals, alpha=1, thresh=1e-6, iterations=300):
        signals = signals.reshape(signals.shape[0], -1)
        m, n = signals.shape
        W = np.random.rand(m, m)

        for c in range(m):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):
                ws = np.dot(w.T, signals)
                wg = np.tanh(ws * alpha).T
                wg_ = (1 - np.square(np.tanh(ws))) * alpha
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())
                lim = np.abs(np.abs((wNew * w).sum()) - 1)
                w = wNew
                i += 1

            W[c, :] = w.T
        return W

    def fit_transform(self, X):
        Xc, meanX = self.center(X)
        Xw, whiteM = self.whiten(Xc)
        W = self.fastICA(Xw, alpha=1)
        unMixed = Xw.T.dot(W.T)
        
        meanX = np.transpose(meanX, (1, 0, 2))

        unMixed = (unMixed.T - meanX).T

        return unMixed


ica = ICA()

X = np.random.rand(360, 4, 250)
X = np.transpose(X, (1, 2, 0))

unMixed = ica.fit_transform(X)