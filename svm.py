import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from kernel_private import KernelEquations
solvers.options['show_progress'] = False

class SVM:

    def __init__(self, kernel, C=0.1):
        self.k = KernelEquations( kernel )
        self.C = C
        
    def fit(self, X, y):
        m, n = X.shape

        K = self.k.calculate_kernel( X = X, gamma = 1, var = 0.1, d = 3, coef0 = 1 )

        P = matrix( np.matmul( y, y.T ) * K )
        q = matrix( np.ones( ( m, 1 ) ) * -1 )

        A = matrix( ( y.reshape( 1, -1 ) ) )
        self.b = matrix( np.zeros( 1 ) )

        G = matrix( np.vstack( ( np.eye( m ) * -1, np.eye( m ) ) ) )
        h = matrix( np.hstack( ( np.zeros( m ), np.ones( m ) * self.C ) ) )

        sol = solvers.qp( P, q, G, h, A, self.b )
        alphas = np.array( sol[ 'x' ] )

        self.w = np.dot( ( y * alphas ).T, X )[ 0 ]
        S = ( alphas > 1e-5 ).flatten()
        self.b = np.mean( y[ S ] - np.dot( X[ S ], self.w.reshape( -1, 1 ) ) )

    def predict(self, X):
        N = X.shape[ 0 ]
        prod = X @ self.w + np.full( N, self.b )
        return np.sign( prod )
