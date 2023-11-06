import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from kernel_private import KernelEquations
solvers.options['show_progress'] = False

class SVM:

    def __init__( self, kernel, C = 0.1 ):
        self.k = KernelEquations( kernel )
        self.C = C
        
    def fit(self, X, y):
        Y = np.diag( y.flatten() )
        N, m = X.shape

        K = np.zeros( ( N,N ) )
        for i in range( N ):
            for j in range( N ):
                K[ i, j ] = self.k.calculate_kernel( X = X[ i, : ], y = X[ j, : ], sigma = 1, var = 0.1, d = 3, coef0 = 1, C = self.C )

        P = matrix( Y @ K @ Y )
        q = matrix( np.ones( ( N,1 ) ) * -1 )
        G = matrix( np.vstack( ( y.T,
                              -1 * y.T,
                              -1 * np.eye( N ),
                              np.eye( N ) ) ) )
        h = matrix( np.vstack( ( np.zeros( ( N + 2, 1 ) ),
                              self.C * np.ones( ( N, 1 ) ) ) ) )
        
        while True:
            try:
                solution = solvers.qp( P, q, G, h )
                break
            
            except:
                pass

        alphas = np.array( solution[ 'x' ] )

        self.w = np.dot( ( alphas * y ).T, X )
        self.b = np.median( np.array( [ np.abs( y[ n, : ] - self.k.calculate_kernel( X = self.w, y = X[ n, : ], C = self.C, sigma = 1, var = 0.1, d = 3, coef0 = 1 ) ) for n in range( N ) ] ) )

    def predict( self, X ):
        N = X.shape[ 0 ]
        output = X @ self.w.T + np.ones( ( N, 1 ) ) * self.b
        for i in range( len( output ) ):
            if output[ i ][ 0 ] < 0:
                output[ i ][ 0 ] = -1

            if output[ i ][ 0 ] > 0:
                output[ i ][ 0 ] = 1

        return output
