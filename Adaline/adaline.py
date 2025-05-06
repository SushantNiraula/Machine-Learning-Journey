import numpy as np
class Adaline:
    """ADAptive LInear NEuron classifier

    Parameters:
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
    Number of iterations
    random_state: int
        Random number generator seed for random weight initialization.

    Attributes:
        w_ : 1d-array
        Weight vector after fitting.
        b_ : scalar
        Bias vector after fitting.
        losses_ : list
        Mean squared error loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state= random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters:
        ------------
        X: array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and
        n_features  is the number of features.
        y: array-like, shape (n_samples,)
        Target values.

        Returns:
        --------
        self: object

        """
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) ## X.shape[1] to match no of weights to no of features.
        self.b_=np.float64(0.)
        self.losses_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(X) ## z=X1xW1+X2xW2+.....+b
            output= self.activation(net_input) ## i.e returns net_input*1 itself.
            errors=(y-output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) /X.shape[0]
            self.b_ += self.eta * 2.0 *errors.mean()
            self.losses_.append(errors)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_) + self.b_
    def activation(self,X):
        return X
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>= 0.5, 1, 0)

