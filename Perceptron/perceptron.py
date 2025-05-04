
import numpy as np
class Perceptron():
    """A simple Perceptron implementation for binary classification.
        parameters
        ----------
        lr=float
            Learning rate for weight updates.(0.0-1.0)
        n_iter= int
            Number of iterations for training.
        random_state =int
            Random state for weight initialization.
        
        Attributes
        ----------
        w_ : 1d-array
            Weights after fitting.
        b_ : scalar
            Bias after fitting.
        
        errors_ : list
            Number of misclassifications in each iteration.    
    
    """
    def __init__(self, lr=0.01, n_iter=1000, random_state=1):
        self.lr= lr
        self.n_iter= n_iter
        self.random_state= random_state

    def fit(self, X,y):
        """ Fit the trainig data to get the weights and bias.
        
        Parameters:
        ----------
        X: {array-like},shape=[n_samples, n_features]
            Training data.
        y: {array-like},shape=[n_samples]
            Target values.
        Returns:
        -------
        self: object
            Returns self.
        """
        rgen= np.random.RandomState(self.random_state) # random state for reproducibility
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # weights where normal distribution
        self.b_=0.0
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(X,y): ## xi is a single sample and target is the corresponding label
                update = self.lr *(target -self.predict(xi) ) # update is the difference between the target and the predicted value
                self.w_ += update * xi # update the weights
                self.b_ += update # update the bias
                errors += int(update != 0.0) # count the number of misclassifications i.e if the update is not 0 then it is a misclassification
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate the net input.
        
        Parameters:
        ----------
        X: {array-like},shape=[n_samples, n_features]
            Training data.
        
        Returns:
        -------
        net_input: scalar
            Net input.
        """
        return np.dot(X, self.w_) + self.b_
    def predict(self, X):
        """Return class label after
        unit step function."""

        return np.where(self.net_input(X)>=0.0, 1, 0) # if net input is greater than or equal to 0 then return 1 else return -1