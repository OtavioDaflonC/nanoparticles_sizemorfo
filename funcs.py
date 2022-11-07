
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder





# para classificação:



def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:

    ohe = OneHotEncoder(sparse=False)
    ohe_cols = X.select_dtypes(include='object').columns.tolist() + ['agencia']
    data_ohe = ohe.fit_transform(X[ohe_cols])
    data_ohe = pd.DataFrame(data_ohe, columns=ohe.get_feature_names_out(), dtype=int)

    new_X = pd.concat([X, data_ohe], axis=1).drop(columns=ohe_cols)

    return new_X