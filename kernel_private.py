from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np

class KernelEquations:
    def __init__( self, kernel_selection = "linear" ):
        self._kernel_selection = kernel_selection

        if kernel_selection == "linear":
            self._kernel_function = lambda X: np.matmul(X, X.T)
        
        elif kernel_selection == "rbf":
            def rbf( X, var, gamma ):
                X_norm = np.sum(X ** 2, axis = -1)
                return var * np.exp( -gamma * ( X_norm[ :,None ] + X_norm[ None, : ] - 2 * np.dot( X, X.T ) ) )
            
            self._kernel_function = rbf

        elif kernel_selection == "poly":
            self._kernel_function = lambda X, d: np.linalg.matrix_power( np.matmul( X, X.T ), d)

        elif kernel_selection == "sigmoid":
            self._kernel_function = lambda X, gamma, coef0: np.tanh( gamma * np.dot( X, X.T ) + coef0 )
    
    def calculate_kernel( self, **kwargs ):
        if self._kernel_selection == "linear":
            return self._kernel_function( kwargs[ "X" ] )

        elif self._kernel_selection == "rbf":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "gamma" ], kwargs[ "var" ]  )
        
        elif self._kernel_selection == "poly":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "d" ] )
        
        elif self._kernel_selection == "sigmoid":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "gamma" ], kwargs[ "coef0" ] )
