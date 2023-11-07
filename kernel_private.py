from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np

class KernelEquations:
    def __init__( self, kernel_selection = "linear" ):
        self._kernel_selection = kernel_selection

        if kernel_selection == "linear":
            self._kernel_function = lambda X, y: np.matmul( X, y.T )
        
        elif kernel_selection == "rbf":
            self._kernel_function = lambda X, y, sigma: np.exp( -( X - y ).T @ ( X - y ) / sigma ** 2 )

        elif kernel_selection == "poly":
            self._kernel_function = lambda X, y, d, C: ( ( X @ y.T ) + C ) ** d

        elif kernel_selection == "sigmoid":
            self._kernel_function = lambda X, y , sigma, coef0: np.tanh( sigma * np.dot( X, y.T ) + coef0 )
    
    def calculate_kernel( self, **kwargs ):
        if self._kernel_selection == "linear":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "y" ] )

        elif self._kernel_selection == "rbf":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "y" ], kwargs[ "sigma" ] )
        
        elif self._kernel_selection == "poly":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "y" ], kwargs[ "d" ], kwargs[ "C" ] )
        
        elif self._kernel_selection == "sigmoid":
            return self._kernel_function( kwargs[ "X" ], kwargs[ "y" ], kwargs[ "sigma" ], kwargs[ "coef0" ] )
