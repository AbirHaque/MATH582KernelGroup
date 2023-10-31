"""
Math 582 - Private Module for Feature Selection
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class FeatureSelection:
    def __init__( self, model, dataframe ):
        self._model = model
        self._dataframe = dataframe
        self._mutation_buffer = []

    def _create_features( self ):
        pca = self._pca()
        pca_matrix=pca.transform(self._dataframe)
        eigenvectors=pca.components_
        eigenvalues=pca.explained_variance_
        feature_vectors=pca_matrix@eigenvectors.T
        return feature_vectors

    def _pca( self ):
        num_cols=self._dataframe.shape[1]
        pca = PCA(n_components = num_cols)
        pca.fit(self._dataframe)
        return pca

    def __create_mutations( self ):
        pass

    def _genetic_algorithm( self ):
        pass

    def _get_best_model( self ):
        pass


