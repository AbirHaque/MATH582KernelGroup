import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class SVM:

    def __init__(self, kernel_fn, C=(10 ** -3)):
        self.k = lambda p, n: p @ n
        self.C = C
        
    def fit(self, X, y):
        m, n = X.shape
        K = np.matmul( X, X.T )

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
