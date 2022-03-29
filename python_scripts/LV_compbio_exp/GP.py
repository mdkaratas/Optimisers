#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:52:10 2022

@author: melikedila
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline



import numpy.matlib
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import cho_solve
from pyDOE import lhs


class GaussianProcess:
    """A class that trains a Gaussian Process model
    to approximate functions"""

    def __init__(self, n_restarts, opt, inital_point,
    kernel, trend, nugget):
        """Initialize a Gaussian Process model
        Input
        ------
        n_restarts (int): number of restarts of the local optimizer
        opt(dict): specify optimization parameters
                   (see scipy.optimize.minimize methods)
                   {'optimizer': str, 'jac': bool}
        inital_point (array): user-specified starting points
        kernel (string): kernel type
        nugget (float): nugget term"""

        self.n_restarts = n_restarts
        self.opt = opt
        self.init_point = inital_point
        self.kernel = kernel
        self.trend = trend
        self.nugget = nugget

    def Corr(self, X1, X2, theta):
        """Construct the correlation matrix between X1 and X2
        based on specified kernel function
        Input
        -----
        X1, X2 (2D array): shape (n_samples, n_features)
        theta (array): correlation legnths for different dimensions
        Output
        ------
        K: the correlation matrix
        """

        # Initialize correlation matrix
        K = np.zeros((X1.shape[0], X2.shape[0]))

        # Compute entries of the correlation matrix
        if self.kernel == 'Gaussian':
            # Gaussian kernel
            for i in range(X1.shape[0]):
                K[i,:] = np.exp(-np.sum(theta*(X1[i,:]-X2)**2, axis=1))

        elif self.kernel == 'Matern-3_2':
            # Matern-3/2 kernel
            for i in range(X1.shape[0]):
                comp = np.sqrt(3)*theta*np.abs(X1[i,:]-X2)
                K[i,:] = np.prod(1+comp, axis=1)*np.exp(-np.sum(comp, axis=1))

        elif self.kernel == 'Matern-5_2':
            # Matern-5/2 kernel
            for i in range(X1.shape[0]):
                comp = np.sqrt(5)*theta*np.abs(X1[i,:]-X2)
                K[i,:] = np.prod(1+comp+comp**2/3, axis=1)*np.exp(-np.sum(comp, axis=1))
        elif self.kernel == 'Cubic':
            # Cubic kernel
            for i in range(X1.shape[0]):
                comp = np.zeros_like(X2)
                diff = theta*np.abs(X1[i,:]-X2)
                # Filter values - first condition
                bool_table = (diff<1) & (diff>0.2)
                comp[bool_table] = 1.25*(1-diff[bool_table])**3
                # Filter values - second condition
                bool_table = (diff<=0.2) & (diff>=0)
                comp[bool_table] = 1-15*diff[bool_table]**2+30*diff[bool_table]**3
                # Construct kernel matrix
                K[i,:] = np.prod(comp, axis=1)

        return K
    
    
    



class GPInterpolator(GaussianProcess):
    """A class that trains a Gaussian Process model
    to interpolate functions"""

    def __init__(self, n_restarts=20, opt={'optimizer':'L-BFGS-B',
    'jac': True}, inital_point=None, verbose=False,
    kernel='Gaussian', trend='Const', nugget=1e-10):

        # Display optimization log
        self.verbose = verbose

        super().__init__(n_restarts, opt, inital_point,
        kernel, trend, nugget)

    def Neglikelihood(self, theta):
        """Negative log-likelihood function
        Input
        -----
        theta (array): correlation legnths for different dimensions
        Output
        ------
        NegLnLike: Negative log-likelihood value
        NegLnLikeDev (optional): Derivatives of NegLnLike"""

        theta = 10**theta    # Correlation length
        n = self.X.shape[0]  # Number of training instances

        if isinstance(self.trend, str):
            if self.trend == 'Const':
                F = np.ones((n,1))
            elif self.trend == 'Linear':
                F = np.hstack((np.ones((n,1)), self.X))
            elif self.trend == 'Quadratic':
                # Problem dimensionality
                dim = self.X.shape[1]
                # Initialize F matrix
                F = np.ones((n,1))
                # Fill in linear part
                F = np.hstack((F, self.X))
                # Fill in quadratic part
                for i in range(dim):
                        F = np.hstack((F, self.X[:, [i]]*self.X[:,i:]))
        else:
            F = self.trend


        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*self.nugget
        L = np.linalg.cholesky(K)

        # Mean estimation
        mu = np.linalg.solve(F.T @ (cho_solve((L, True), F)),
        F.T @ (cho_solve((L, True), self.y)))
        # mu = (F.T @ (cho_solve((L, True), self.y))) / \
            # (F.T @ (cho_solve((L, True), F)))

        # Variance estimation
        SigmaSqr = (self.y-F@mu).T @ (cho_solve((L, True), self.y-F@mu)) / n

        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        NegLnLike = (n/2)*np.log(SigmaSqr) + 0.5*LnDetK

        # Update attributes
        self.K, self.F, self.L, self.mu, self.SigmaSqr = K, F, L, mu, SigmaSqr

        # If derivatives are not calculated
        if self.opt['jac'] is False:

            return NegLnLike.flatten()

        # If derivatives are calculated
        else:

            # Compute derivative of log-likelihood (adjoint)
            # 1-Construct adjoint kernel matrix
            adjoint_K = 1/(2*SigmaSqr)*((cho_solve((L, True), self.y-F@mu)) @
            (cho_solve((L, True), self.y-F@mu)).T) - 0.5*(cho_solve((L, True), np.eye(n)))

            K_combo = K*adjoint_K

            # 2-Calculate derivatives
            total_sum = np.zeros(self.X.shape[1])

            for i in range(self.X.shape[1]):
                broadcast = (np.matlib.repmat(self.X[:,[i]],1,n)-
                np.matlib.repmat(self.X[:,[i]].T,n,1))**2
                total_sum[i] = np.concatenate(broadcast*K_combo).sum()

            NegLnLikeDev = np.log(10)*theta*total_sum

            return NegLnLike.flatten(), NegLnLikeDev.flatten()

    def fit(self, X, y):
        """GP model training
        Input
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        """

        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub-lb)*lhd + lb

        # Expand initial points if user specified them
        if self.init_point is not None:
            initial_points = np.vstack((initial_points, self.init_point))

        # Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros(self.n_restarts)
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood,
            initial_points[i,:],
            jac=self.opt['jac'],
            method=self.opt['optimizer'],
            bounds=bnds)

            opt_para[i,:] = res.x
            opt_func[i] = res.fun

            # Display optimization progress in real-time
            if self.verbose == True:
                print('Iteration {}: Likelihood={} \n'
                .format(str(i+1), np.min(opt_func[:i+1])))

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        if self.opt['jac'] is False:
            self.NegLnlike = self.Neglikelihood(self.theta)
        else:
            self.NegLnlike, self.NegLnLikeDev = self.Neglikelihood(self.theta)

    def predict_only(self, X, y, theta):
        """Predict-only mode, with given theta value
        Input:
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        theta: (array): correlation legnths for different dimensions"""

        # Update training data
        self.X, self.y = X, y

        # Update attributes
        self.theta = theta

        if self.opt['jac'] is False:
            self.NegLnlike = self.Neglikelihood(self.theta)
        else:
            self.NegLnlike, self.NegLnLikeDev = self.Neglikelihood(self.theta)

    def predict(self, X_test, trend=None, cov_return=False):
        """GP model predicting
        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        trend: trend values at test sites, shape (n_samples, n_functions)
        cov_return (bool): return/not return covariance matrix
        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)

        # Mean prediction
        n = X_test.shape[0]  # Number of training instances
        dim = X_test.shape[1]  # Problem dimension

        if isinstance(self.trend, str):
            if self.trend == 'Const':
                f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))
            elif self.trend == 'Linear':
                obs = np.hstack((np.ones((n,1)), X_test))
                f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))
            elif self.trend == 'Quadratic':
                obs = np.ones((n,1))
                obs = np.hstack((obs, X_test))
                for i in range(dim):
                        obs = np.hstack((obs, X_test[:, [i]]*X_test[:,i:]))
                f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))

        else:
            f = trend @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))


        # Variance prediction
        SSqr = self.SigmaSqr*(1 - np.diag(k.T @ (cho_solve((self.L, True), k))))

        # Calculate covariance
        if cov_return is True:
            Cov = self.SigmaSqr*(self.Corr(X_test, X_test, 10**self.theta)
             - k.T @ (cho_solve((self.L, True), k)))

            # Return values
            return f.flatten(), SSqr.flatten(), Cov

        else:
            # Return values
            return f.flatten(), SSqr.flatten()

    def get_params(self, deep = False):
        return {'n_restarts':self.n_restarts, 'opt': self.opt,
        'inital_point': self.init_point, 'verbose': self.verbose,
        'kernel': self.kernel, 'trend': self.trend, 'nugget': self.nugget}

    def score(self, X_test, y_test, trend=None):
        """Calculate root mean squared error
        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        y_test (array): test labels
        trend: trend values at test sites, shape (n_samples, n_functions)
        Output
        ------
        RMSE: the root mean square error"""

        y_pred, SSqr = self.predict(X_test, trend)
        RMSE = np.sqrt(np.mean((y_pred-y_test.flatten())**2))

        return RMSE

    def LOOCV(self):
        """Calculate leave-one-out cross-validation error
        Approximation algorithm is used speed up the calculation, see
        [Ref]Predictive approaches for choosing hyperparameters in
        Gaussian processes. Neural Comput. 13, 1103–1118
        (https://www.mitpressjournals.org/doi/abs/10.1162/08997660151134343)
        Output:
        ------
        LOO (array): Leave-one-out cross validation error at each training location
        e_CV: mean squared LOOCV error
        """

        # Calculate CV error
        Q = cho_solve((self.L, True), self.y-self.F@self.mu)
        LOO = Q.flatten()/np.diag(cho_solve((self.L, True), np.eye(self.X.shape[0])))

        e_CV = np.sqrt(np.mean(LOO**2))

        return e_CV, LOO

    def enrichment(self, criterion, candidate, diagnose=False):
        """Training sample enrichment for active learning
        Input:
        ------
        criterion (dict): learning criterion
        candidate (array): candidate sample pool, shape (n_samples, n_features)
        Output:
        -------
        target (float): the optimum target value
        index (array): the index of the selected sample
        orignal_pool (array): original sample pool
        reduced_pool (array): reduced sample pool (remove the selected sample)
        diagnostics (array): optional, the array of diagnostic results"""

        if criterion['Condition'] == 'EPE':

            # Compute cross-validation error
            LOO = self.LOOCV()[1]

            # Compute prediction variance
            pred, pred_var = self.predict(candidate)

            # Calculate bias
            bias = np.zeros(candidate.shape[0])
            for i in range(candidate.shape[0]):
                # Determine bias
                distance_sqr = np.sum((candidate[[i],:]-self.X)**2, axis=1)
                closest_index = np.argmin(distance_sqr.flatten())
                bias[i] = LOO[closest_index]**2

            # Calculate expected prediction error
            expected_error = bias + pred_var
            target = np.max(expected_error)

            # Locate promising sample
            index = np.argmax(expected_error)

            # Select promising sample
            sample = candidate[[index],:]
            reduced_candidate = np.delete(candidate, obj=index, axis=0)

            # For diagnose purposes
            diagnostics = expected_error

        elif criterion['Condition'] == 'U':

            # Make predictions
            pred, pred_var = self.predict(candidate)

            # Calculate U values
            U_values = np.abs(pred-criterion['Threshold'])/np.sqrt(pred_var)
            target = np.min(U_values)

            # Locate promising sample
            index = np.argmin(U_values)

            # Select promising sample
            sample = candidate[[index],:]
            reduced_candidate = np.delete(candidate, obj=index, axis=0)

            # For diagnose purposes
            diagnostics = U_values

        if diagnose is True:
            return target, sample, candidate, reduced_candidate, diagnostics
        else:
            return target, sample, candidate, reduced_candidate

    def realizations(self, N, X_eval):
        """Draw realizations from posterior distribution of
        the trained GP metamodeling
        Input:
        -----
        N: Number of realizations
        X_eval: Evaluate coordinates
        Output:
        -------
        samples: Generated realizations, shape (N, n_features)"""

        f, SSqr, Cov = self.predict(X_eval, cov_return=True)
        Cov = (Cov + Cov.T)/2

        samples = np.random.default_rng().multivariate_normal(mean=f, cov=Cov, size=N)

        return samples






class GPInterpolator(GaussianProcess):
    """A class that trains a Gaussian Process model
    to interpolate functions"""

    def __init__(self, n_restarts=20, opt={'optimizer':'L-BFGS-B',
    'jac': True}, inital_point=None, verbose=False,
    kernel='Gaussian', trend='Const', nugget=1e-10):

        # Display optimization log
        self.verbose = verbose

        super().__init__(n_restarts, opt, inital_point,
        kernel, trend, nugget)

    def Neglikelihood(self, theta):
        """Negative log-likelihood function
        Input
        -----
        theta (array): correlation legnths for different dimensions
        Output
        ------
        NegLnLike: Negative log-likelihood value
        NegLnLikeDev (optional): Derivatives of NegLnLike"""

        theta = 10**theta    # Correlation length
        n = self.X.shape[0]  # Number of training instances

        if isinstance(self.trend, str):
            if self.trend == 'Const':
                F = np.ones((n,1))
            elif self.trend == 'Linear':
                F = np.hstack((np.ones((n,1)), self.X))
            elif self.trend == 'Quadratic':
                # Problem dimensionality
                dim = self.X.shape[1]
                # Initialize F matrix
                F = np.ones((n,1))
                # Fill in linear part
                F = np.hstack((F, self.X))
                # Fill in quadratic part
                for i in range(dim):
                        F = np.hstack((F, self.X[:, [i]]*self.X[:,i:]))
        else:
            F = self.trend


        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*self.nugget
        L = np.linalg.cholesky(K)

        # Mean estimation
        mu = np.linalg.solve(F.T @ (cho_solve((L, True), F)),
        F.T @ (cho_solve((L, True), self.y)))
        # mu = (F.T @ (cho_solve((L, True), self.y))) / \
            # (F.T @ (cho_solve((L, True), F)))

        # Variance estimation
        SigmaSqr = (self.y-F@mu).T @ (cho_solve((L, True), self.y-F@mu)) / n

        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        NegLnLike = (n/2)*np.log(SigmaSqr) + 0.5*LnDetK

        # Update attributes
        self.K, self.F, self.L, self.mu, self.SigmaSqr = K, F, L, mu, SigmaSqr

        # If derivatives are not calculated
        if self.opt['jac'] is False:

            return NegLnLike.flatten()

        # If derivatives are calculated
        else:

            # Compute derivative of log-likelihood (adjoint)
            # 1-Construct adjoint kernel matrix
            adjoint_K = 1/(2*SigmaSqr)*((cho_solve((L, True), self.y-F@mu)) @
            (cho_solve((L, True), self.y-F@mu)).T) - 0.5*(cho_solve((L, True), np.eye(n)))

            K_combo = K*adjoint_K

            # 2-Calculate derivatives
            total_sum = np.zeros(self.X.shape[1])

            for i in range(self.X.shape[1]):
                broadcast = (np.matlib.repmat(self.X[:,[i]],1,n)-
                np.matlib.repmat(self.X[:,[i]].T,n,1))**2
                total_sum[i] = np.concatenate(broadcast*K_combo).sum()

            NegLnLikeDev = np.log(10)*theta*total_sum

            return NegLnLike.flatten(), NegLnLikeDev.flatten()

    def fit(self, X, y):
        """GP model training
        Input
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        """

        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub-lb)*lhd + lb

        # Expand initial points if user specified them
        if self.init_point is not None:
            initial_points = np.vstack((initial_points, self.init_point))

        # Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros(self.n_restarts)
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood,
            initial_points[i,:],
            jac=self.opt['jac'],
            method=self.opt['optimizer'],
            bounds=bnds)

            opt_para[i,:] = res.x
            opt_func[i] = res.fun

            # Display optimization progress in real-time
            if self.verbose == True:
                print('Iteration {}: Likelihood={} \n'
                .format(str(i+1), np.min(opt_func[:i+1])))

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        if self.opt['jac'] is False:
            self.NegLnlike = self.Neglikelihood(self.theta)
        else:
            self.NegLnlike, self.NegLnLikeDev = self.Neglikelihood(self.theta)

    def predict_only(self, X, y, theta):
        """Predict-only mode, with given theta value
        Input:
        -----
        X (array): shape (n_samples, n_features)
        y (array): shape (n_samples, 1)
        theta: (array): correlation legnths for different dimensions"""

        # Update training data
        self.X, self.y = X, y

        # Update attributes
        self.theta = theta

        if self.opt['jac'] is False:
            self.NegLnlike = self.Neglikelihood(self.theta)
        else:
            self.NegLnlike, self.NegLnLikeDev = self.Neglikelihood(self.theta)

    def predict(self, X_test, trend=None, cov_return=False):
        """GP model predicting
        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        trend: trend values at test sites, shape (n_samples, n_functions)
        cov_return (bool): return/not return covariance matrix
        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)

        # Mean prediction
        n = X_test.shape[0]  # Number of training instances
        dim = X_test.shape[1]  # Problem dimension

        if isinstance(self.trend, str):
            if self.trend == 'Const':
                f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))
            elif self.trend == 'Linear':
                obs = np.hstack((np.ones((n,1)), X_test))
                f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))
            elif self.trend == 'Quadratic':
                obs = np.ones((n,1))
                obs = np.hstack((obs, X_test))
                for i in range(dim):
                        obs = np.hstack((obs, X_test[:, [i]]*X_test[:,i:]))
                f = obs @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))

        else:
            f = trend @ self.mu + k.T @ (cho_solve((self.L, True), self.y-self.F@self.mu))


        # Variance prediction
        SSqr = self.SigmaSqr*(1 - np.diag(k.T @ (cho_solve((self.L, True), k))))

        # Calculate covariance
        if cov_return is True:
            Cov = self.SigmaSqr*(self.Corr(X_test, X_test, 10**self.theta)
             - k.T @ (cho_solve((self.L, True), k)))

            # Return values
            return f.flatten(), SSqr.flatten(), Cov

        else:
            # Return values
            return f.flatten(), SSqr.flatten()

    def get_params(self, deep = False):
        return {'n_restarts':self.n_restarts, 'opt': self.opt,
        'inital_point': self.init_point, 'verbose': self.verbose,
        'kernel': self.kernel, 'trend': self.trend, 'nugget': self.nugget}

    def score(self, X_test, y_test, trend=None):
        """Calculate root mean squared error
        Input
        -----
        X_test (array): test set, shape (n_samples, n_features)
        y_test (array): test labels
        trend: trend values at test sites, shape (n_samples, n_functions)
        Output
        ------
        RMSE: the root mean square error"""

        y_pred, SSqr = self.predict(X_test, trend)
        RMSE = np.sqrt(np.mean((y_pred-y_test.flatten())**2))

        return RMSE

    def LOOCV(self):
        """Calculate leave-one-out cross-validation error
        Approximation algorithm is used speed up the calculation, see
        [Ref]Predictive approaches for choosing hyperparameters in
        Gaussian processes. Neural Comput. 13, 1103–1118
        (https://www.mitpressjournals.org/doi/abs/10.1162/08997660151134343)
        Output:
        ------
        LOO (array): Leave-one-out cross validation error at each training location
        e_CV: mean squared LOOCV error
        """

        # Calculate CV error
        Q = cho_solve((self.L, True), self.y-self.F@self.mu)
        LOO = Q.flatten()/np.diag(cho_solve((self.L, True), np.eye(self.X.shape[0])))

        e_CV = np.sqrt(np.mean(LOO**2))

        return e_CV, LOO

    def enrichment(self, criterion, candidate, diagnose=False):
        """Training sample enrichment for active learning
        Input:
        ------
        criterion (dict): learning criterion
        candidate (array): candidate sample pool, shape (n_samples, n_features)
        Output:
        -------
        target (float): the optimum target value
        index (array): the index of the selected sample
        orignal_pool (array): original sample pool
        reduced_pool (array): reduced sample pool (remove the selected sample)
        diagnostics (array): optional, the array of diagnostic results"""

        if criterion['Condition'] == 'EPE':

            # Compute cross-validation error
            LOO = self.LOOCV()[1]

            # Compute prediction variance
            pred, pred_var = self.predict(candidate)

            # Calculate bias
            bias = np.zeros(candidate.shape[0])
            for i in range(candidate.shape[0]):
                # Determine bias
                distance_sqr = np.sum((candidate[[i],:]-self.X)**2, axis=1)
                closest_index = np.argmin(distance_sqr.flatten())
                bias[i] = LOO[closest_index]**2

            # Calculate expected prediction error
            expected_error = bias + pred_var
            target = np.max(expected_error)

            # Locate promising sample
            index = np.argmax(expected_error)

            # Select promising sample
            sample = candidate[[index],:]
            reduced_candidate = np.delete(candidate, obj=index, axis=0)

            # For diagnose purposes
            diagnostics = expected_error

        elif criterion['Condition'] == 'U':

            # Make predictions
            pred, pred_var = self.predict(candidate)

            # Calculate U values
            U_values = np.abs(pred-criterion['Threshold'])/np.sqrt(pred_var)
            target = np.min(U_values)

            # Locate promising sample
            index = np.argmin(U_values)

            # Select promising sample
            sample = candidate[[index],:]
            reduced_candidate = np.delete(candidate, obj=index, axis=0)

            # For diagnose purposes
            diagnostics = U_values

        if diagnose is True:
            return target, sample, candidate, reduced_candidate, diagnostics
        else:
            return target, sample, candidate, reduced_candidate

    def realizations(self, N, X_eval):
        """Draw realizations from posterior distribution of
        the trained GP metamodeling
        Input:
        -----
        N: Number of realizations
        X_eval: Evaluate coordinates
        Output:
        -------
        samples: Generated realizations, shape (N, n_features)"""

        f, SSqr, Cov = self.predict(X_eval, cov_return=True)
        Cov = (Cov + Cov.T)/2

        samples = np.random.default_rng().multivariate_normal(mean=f, cov=Cov, size=N)

        return samples





import matlab.engine
import numpy as np
import cma
import pickle
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem
from pynmmso.listeners import TraceListener

eng = matlab.engine.start_matlab()


#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataDD.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingDD.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataDD']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingDD']



#%%

#####################################################      CMA-ES: G x G  01

def neuro1lp_gates(inputparams,gates):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost


# Training data
X_train = np.array([0.0, 0.15, 0.3, 0.54, 0.7, 0.8, 1]).reshape(-1,1)
y_train = neuro1lp_gates(X_train,[0,1])

#init_sol = []
for i in range(6):
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    u = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u]

    X_train = np.array(init_sol)
    y_train = neuro1lp_gates(X_train,[0,1])
    
    
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    u = np.random.uniform(0,1)
    test_sol = [x,y,z,t,u]
    
    
    # Testing data
    X_test = np.array(test_sol)
    y_test = neuro1lp_gates(test_sol,[0,1])


# Training data
sample_num = 30
lb, ub = np.array([-2, -1]), np.array([2, 3])
X_train = (ub-lb)*lhs(2, samples=sample_num) + lb

# Compute labels
y_train = Test_2D(X_train).reshape(-1,1)

# Test data
X1 = np.linspace(-2, 2, 20)
X2 = np.linspace(-1, 3, 20)
X1, X2 = np.meshgrid(X1, X2)
X_test = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
y_test = Test_2D(X_test)




sample_num = 30
lb = np.array([0,0,0,0,0], dtype='float')
ub = np.array([24,24,12,1,1] ,dtype='float')
X_train = (ub-lb)*lhs(5, samples=sample_num) + lb

y_train = neuro1lp_gates(X_train,[0,1]).reshape(-1,1)

# Test data
X1 = np.linspace(0, 24, 20)
X2 = np.linspace(0, 24, 20)
X3 = np.linspace(0, 12, 20)
X4 = np.linspace(0, 1, 20)
X5 = np.linspace(0, 1, 20)
X1, X2, X3, X4, X5 = np.meshgrid(X1,X2,X3,X4,X5)
X_test = list(np.hstack((X1.reshape(-1,1), X2.reshape(-1,1), X3.reshape(-1,1), X4.reshape(-1,1), X5.reshape(-1,1))))
y_test = neuro1lp_gates(X_test,[0,1])



X_test[0:2]

























