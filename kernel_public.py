from feature_private import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def main():
    wkspc()

def wkspc():
    dataset = pd.redatasetad_csv('DiabetesBinaryClassification.csv')
    FeatureSelection(None, dataset )