import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


#Feature Selection
def get_corrs(df, subset = 'persona_id', column = 'target', show_viz = False):
    '''
    get_corrs(df, subset = 'persona_id', column = 'target', show_viz = False)

    Use this function to run t-tests on all categories within a given feature on some target variable, returns crosstab matrix of p_values

    args: 
    df: pandas dataframe
    subset: name of categorical column for which you want to compare values in
    column: target column that you wish to test on subsets
    show_viz: if True, will output matrix as heatmap
    '''
    # init blank dictionaries for building dataframe
    ttests = {}
    row = {}
    # compare every combination of two unique values in subset column
    for i in df[subset].unique():
        for j in df[subset].unique():
            # run t-test
            ttests[j] = ttest_ind(df[df[subset] == i][column],df[df[subset] == j][column], nan_policy='omit',)[1]
        row[i] = ttests.copy()
    if show_viz == True:
        sns.heatmap(pd.DataFrame(row))
        plt.show()
    return pd.DataFrame(row)

        











        