"""
Math 582 - Private Module for Feature Selection

Sources:
    Manoj Turaga's EECS 658 Assignment
"""
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
from svm import SVM

from random import choice, choices

class FeatureSelection:
    def __init__( self, model, dataframe ):
        self._model = model
        self._dataframe = dataframe
        self._categories = []
        self._vals = dict()
        self._output = ""

        self.__create_features()

    def __create_features( self ):
        new_feature_values = self.__pca_features()
        original_columns = [ "Number of times pregnant","Plasma glucose concentration a 2 hours in an oral glucose tolerance test","Diastolic blood pressure (mm Hg)","Triceps skin fold thickness (mm)","2-Hour serum insulin (mu U/ml)","Body mass index (weight in kg/(height in m)^2)","Diabetes pedigree function","Age (years)","Class variable (0 or 1)" ]
        new_features_columns = [ f'z{ i + 1 }' for i in range( len( self._dataframe.columns ) - 1 ) ]
        
        self._dataframe.columns = original_columns
        concated_data = pd.DataFrame( new_feature_values, columns = new_features_columns )
        self._dataframe = pd.concat( [ concated_data, self._dataframe ], axis = 1 )

        self._categories = self._dataframe.columns[ : -1 ]
        self._output = self._dataframe.columns[ -1 ]

        for column, data in self._dataframe.items():
            self._vals[ column ] = data

        self._vals[ self._output ] = np.asarray( [ np.array( [ -1 if d == 0 else 1 ], dtype = float ) for d in self._vals[ self._output ] ] ).T

    def __pca_features( self ):
        pca = PCA( n_components = len( self._dataframe.columns ) - 1 )
        x = []
        for data in self._dataframe.values:
            x.append( np.array( data[ : -1 ], dtype = float ) )

        x = np.asarray( x )

        pca.fit( x )
        principleComponents = pca.transform( x )

        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        return principleComponents @ eigenvectors.T

    def __create_mutations( self, list_of_subsets ):
        max_modify = 1
        max_action = 3

        modify_amount = 1
        action = choice( [ i for i in range( max_action ) ] )

        mutated_subsets = []

        for subset in list_of_subsets:
            temp_mod_amount = 0

            temp_subset = subset.copy()

            if action == 0:
                while len( temp_subset ) < len( self._categories ) and temp_mod_amount < modify_amount:
                    new_feature = choice( [ feature for feature in self._categories if feature not in temp_subset ] )
                    temp_subset.append( new_feature )
                    temp_mod_amount += 1

                else:
                    while temp_mod_amount < modify_amount:
                        remove_feature = choice( [ feature for feature in temp_subset ] )
                        temp_subset.remove( remove_feature )
                        temp_mod_amount += 1

            elif action == 1:
                while len( temp_subset ) > 0 and temp_mod_amount < modify_amount:
                    remove_feature = choice( [ feature for feature in temp_subset ] )
                    temp_subset.remove( remove_feature )
                    temp_mod_amount += 1

                else:
                    while temp_mod_amount < modify_amount:
                        new_feature = choice( [ feature for feature in self._categories if feature not in temp_subset ] )
                        temp_subset.append( new_feature )
                        temp_mod_amount += 1

            elif action == 2:
                while temp_mod_amount < modify_amount:
                    switch_index = choice( [ i for i in range( len( temp_subset ) ) ] )
                    switch_feature = choice( [ feature for feature in self._categories ] )
                    temp_subset[ switch_index ] = switch_feature
                    temp_subset = list( set( temp_subset ) )
                    temp_mod_amount += 1

            mutated_subsets.append( temp_subset )

        return mutated_subsets

    def __genetic_algorithm( self ):
        best_5 = [ list( self._categories ) ] + [ list( choices( self._categories, k = 5 ) ) for i in range( 5 ) ]        
        best_features = []
        best_accuracy = 0

        for iteration in range( 10 ):
            unions = []
            intersections = []
            list_of_sets = []
#
            completed_combos = []
            for i in range( len( best_5 ) ):
                for j in range( len( best_5 ) ):
                    if i != j and ( i, j ) not in completed_combos and ( j, i ) not in completed_combos:
                        unions.append( list( set( best_5[ i ] ).union( set( best_5[ i ] ) ) ) )
                        intersections.append( list( set( best_5[ i ] ).intersection( set( best_5[ i ] ) ) ) )
                        completed_combos.append( ( i, j ) )
                        completed_combos.append( ( j, i ) )

            list_of_sets.extend( best_5 )
            list_of_sets.extend( unions )
            list_of_sets.extend( intersections )
            list_of_sets.extend( self.__create_mutations( list_of_sets ) )
            
            subset_to_accuracy = []
            for subset in list_of_sets:
                x = np.asarray( [ self._vals[ feature ] for feature in subset ] ).T
                y = np.asarray( self._vals[ self._output ] ).T

                X_Fold1, X_Fold2, Y_Fold1, Y_Fold2 = train_test_split( x,  y , test_size = 0.50, random_state = 1 )
                self._model.fit( X_Fold1, Y_Fold1 )
                pred1 = self._model.predict( X_Fold2 )

                self._model.fit( X_Fold2, Y_Fold2 )
                pred2 = self._model.predict( X_Fold1 )

                pred_outputs = concatenate( [ pred1, pred2 ] )
                true_outputs = concatenate( [ Y_Fold2, Y_Fold1 ] )
                subset_to_accuracy.append( ( subset, accuracy_score( true_outputs, pred_outputs ) ) )

            subset_to_accuracy.sort( key = lambda x: x[ 1 ], reverse = True )
            best_features = subset_to_accuracy[ 0 ][ 0 ]
            best_accuracy = subset_to_accuracy[ 0 ][ 1 ]

            best_5.clear()
            best_5 = [ subset_to_accuracy[ feature ][ 0 ] for feature in range( 5 ) ]
            has_100 = False
            for feature in range( 5 ):
                if subset_to_accuracy[ feature ][ 1 ] == 100: has_100 = True

            if has_100: break

        return best_features, best_accuracy

    def get_best_model( self ):
        best_features, best_accuracy = self.__genetic_algorithm()
        print( best_features )
        print( len( best_features ) )
        print( best_accuracy )

dataframe = read_csv( "DiabetesBinaryClassification.csv")
dataframe = dataframe.drop_duplicates()
dataframe = dataframe.dropna()
temp = FeatureSelection( SVM( "sigmoid" ), dataframe )
temp.get_best_model()


