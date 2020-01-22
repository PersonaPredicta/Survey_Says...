import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it

from scipy import stats
from scipy.stats import ttest_ind


# Recoder Dictionaries
recoder = {
    'r2to2': {
        0: 0,
        1: 1,
    },
    'r4to2': {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    },
    'r5to2lo': {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 1,
    },
    'r5to2hi': {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
    },
    'r5to3': {
        0: 0,
        1: 0,
        2: .5,
        3: 1,
        4: 1,
    },
    'r6to2': {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
    },
    'r6to3': {
        0: 0,
        1: 0,
        2: .5,
        3: .5,
        4: 1,
        5: 1,
    },
}
recoder_dictionaries = {
    2: {
        "important5": 'r5to2hi', # neutral included in no
        "likely5": 'r5to2lo', # neutral included in yes
        "often4": 'r4to2', 
        "often5": 'r5to2lo', # neutral included in yes
        "tenure6": 'r6to2',
    },
    3: {
        "important5": 'r5to3',
        "likely5": 'r5to3',
        "often4": 'r4to2', # three options not practical
        "often5": 'r5to3',
        "tenure6": 'r6to3',            
    },
}


# Add target to dataframe
def add_target_to_df(df):
    learning_conference_int = np.array(df['learning_conference'] >1).astype('int32')
    likely_conference_int = np.array(df['likely_conference']>1).astype('int32')
    df['target'] = (learning_conference_int + likely_conference_int) / 2
    return df

#Feature Selection
def get_corrs(df, subset = 'persona_id', column = 'target', show_viz = False):
    '''
    get_corrs(df, subset = 'persona_id', column = 'target', show_viz = False)

    Use this function to run t-tests on all categories within a given feature 
    on some target variable, returns crosstab matrix of p_values

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
            ttests[j] = ttest_ind(
                df[df[subset] == i][column],df[df[subset] == j][column], nan_policy='omit',
            )[1]
        row[i] = ttests.copy()
    if show_viz == True:
        sns.heatmap(pd.DataFrame(row))
        plt.show()
    return pd.DataFrame(row)

        
# Get mapping dictionary
def get_recoder_dictionary(data_dictionary, recoder_dictionaries=recoder_dictionaries):
    '''
    get_recoder_dictionary(data_dictionary, recoder_dictionaries=recoder_dictionaries)

    Use this function to identify which data dictionary will be used for each 
    field when recoding. One column for each target number of fields.
    
    Activate as follows:
        check_recoder = get_recoder_dictionary(dictdf, recoder_dictionaries)
    

    Returns recoder dictionary dataframe
    '''
    recode_df = data_dictionary[['column_name','encoder']].copy().dropna()
    recode_df['col'] = recode_df.column_name
    recode_df=recode_df.set_index('column_name')
    levels = [level for level in recoder_dictionaries]
    for chk in range(len(levels)):
        level = levels[chk]
        recoder_dictionary = recoder_dictionaries[level]
        recodes = [recode for recode in recoder_dictionary]
        tempdf = recode_df[recode_df.encoder.isin(recodes)].copy()
        tempdf[level] = tempdf.encoder.apply(lambda x: recoder_dictionary[x])
        recode_df = recode_df.join(tempdf[[level]], how='left')
    recode_df = recode_df.dropna(subset=[levels[0]])
    return recode_df


def recode_columns(df, recode_df, recoder=recoder, outputs=2):
    '''
    recode_columns(df, recode_df, recoder=recoder, outputs=2)

    *** MUST HAVE RECODER DICTIONARY FILE BEFORE STARTING ***
    Use this function to make a recoded dataframe. All recode columns will be updated with the new values, other columns will remain as-is.
    Activate as follows:
        dfrecode2 = recode_columns(df, check_recoder, recoder, outputs=2)

    Returns recode dataframe
    '''
    use_df = df.copy()
    allcols=recode_df.col
    check_dicts=recode_df[outputs]
    cols=[chkcol for chkcol in allcols if chkcol in use_df.columns]
    for col in cols:
        use_dict = recoder[check_dicts[col]]
        encoded = use_df[col].apply(lambda x: use_dict[x])
        use_df[col] = encoded
    return use_df


def make_recode_df(
    df, data_dictionary, outputs=2, 
    recoder_dictionaries=recoder_dictionaries, recoder=recoder
    ):
    '''
    make_recode_df(
        df, data_dictionary, outputs=2, 
        recoder_dictionaries=recoder_dictionaries, recoder=recoder
    )

    Function takes wrangle data and dataframe and creates a new dataframe with
    the quantitative values recoded based on the number of outputs
    '''
    check_recoder = get_recoder_dictionary(
        data_dictionary=data_dictionary, 
        recoder_dictionaries=recoder_dictionaries
    )
    dfrecode = recode_columns(
        df=df, recode_df=check_recoder, recoder=recoder, outputs=outputs
    )
    
    return dfrecode


def countplot_cols(df, cols):
    '''
    countplot_cols(df, cols)
    cols = column names in a list
    
    use this function to produce plots showing data distribution by descrete 
    values within a dataframe
    '''
    chk_cols = [col for col in cols if col in df.columns]
    for chk_col in chk_cols:
        plot_col = df[chk_col].value_counts().sort_index()
        use_bins = plot_col.nunique()
        print(f'Column: {chk_col}, {use_bins} unique values')
        print(plot_col)
        plot_col.plot.bar()
        plt.show()


def countplot_tricols(df0, df1, df2, cols):
    '''
    countplot_tricols(df0, df1, df2, cols)
    cols = column names in a list
    
    use this function to produce plots showing data distribution by descrete 
    values across three dataframes
    '''
    cols0 = df0.columns.to_list()
    cols1 = df1.columns.to_list()
    cols2 = df2.columns.to_list()
    chk_col_list = set(cols0 + cols1 + cols2)
    chk_cols = [col for col in cols if col in chk_col_list]
    for chk_col in chk_cols:
        print(chk_col)
        plot0, plot1, plot2, plot_it = False, False, False, False
        if chk_col in cols0:
            plot_col0 = df0[chk_col].astype('category').value_counts().sort_index()
            x0 = plot_col0.index
            y0 = plot_col0.to_list()
            n0 = len(y0)
            plot0 = True
            plot_it=True
        if chk_col in cols1:
            plot_col1 = df1[chk_col].astype('category').value_counts().sort_index()
            x1 = plot_col1.index
            y1 = plot_col1.to_list()
            n1 = len(y1)
            plot1 = True
            plot_it=True
        if chk_col in cols2:
            plot_col2 = df2[chk_col].astype('category').value_counts().sort_index()
            x2 = plot_col2.index
            y2 = plot_col2.to_list()
            n2 = len(y2)
            plot2 = True
            plot_it=True
        if plot_it:
            fig, axs = plt.subplots(
                nrows=1, 
                ncols=3, 
                figsize=(9,3),
                sharey=True, 
            )
            fig.suptitle(chk_col)
            if plot0:
                axs[0].bar(x0,y0,align='center',width=.85, tick_label=x0)
            if plot1:
                axs[1].bar(x1,y1,align='center',width=.45, tick_label=x1)
            if plot2:
                axs[2].bar(x2,y2,align='center',width=.45, tick_label=x2)
            plt.show()
        print('*****')

        
def get_chi2_results(
    chk_df, 
    chk_col, 
    vs_col, 
    display_obs=False,
    print_it=False, 
    return_all=False
):
    '''
    get_chi2_results(
        chk_df, chk_col, vs_col, 
        display_obs=False,
        print_it=False, 
        return_all=False
    )

    Function to perform chi-square test on two categorical fields.
    '''
    if print_it:
        print(f'chk_df  = {chk_df.shape}')
        print(f'chk_col = {chk_df[chk_col].name}')
        print(f'vs_col  = {chk_df[vs_col].name}')
    
    observed = pd.crosstab(index=chk_df[chk_col], columns=chk_df[vs_col])
    if display_obs: # Jupyter notebooks only #
        display(observed) # Jupyter notebooks only #
    chi2, p, degf, expected = stats.chi2_contingency(observed)    
    if print_it:
        print('Observed\n')
        print(observed.values)
        print('---\nExpected\n')
        print(expected)
        print('---\n')
        print(f'degf  = {degf:d}')
        print(f'chi^2 = {chi2:.4f}')
        print(f'p     = {p:.4f}')
    if return_all:
        return chi2, p, degf, expected, observed
    else:
        return p


def get_chi2_for_cols(df, cols, target=None, alpha=.05, print_detail=False, print_sum=False):
    '''
    get_chi2_for_cols(df, cols, target=None, alpha=.05, print_detail=False, print_sum=False)
    
    Function performs chi-square analysis based on a list of categorical 
    columns. If a target is specified, all selected columns will be compared 
    to the target. If not, function will identify all unique combinations of 
    the listed columns.


    df = source dataframe
    cols = list of column names to perform test on
    target = name of target column (OPTIONAL - leave out or specify None to omit)
    print_detail = print all results on screen
    print_sum = print summary on screen after function has run

    EXAMPLE:
        chi2_2df = get_chi2_for_cols(dfrecode2, chi2_cols, 'target', print_sum=True)
    '''
    chi2_df=pd.DataFrame(columns=['chk_col', 'vs_col', 'p_val'])
    hits, atts = 0, 0    
    chk_cols = [col for col in cols if col in df.columns]
    if target:
        iterlist = [(col, target) for col in chk_cols if col != target]
    else:
        iterlist = it.combinations(chk_cols, 2)
    for chk_col, vs_col in iterlist:
        atts += 1
        pval = get_chi2_results(
            chk_df=df, 
            chk_col=chk_col,
            vs_col=vs_col, 
            display_obs=False,
            print_it=False, 
            return_all=False
        )
        if pval < alpha:
            hits +=1
            if print_detail:
                print(f'Hit {hits} of {atts}:')
                print(f'columns = {chk_col} : {vs_col}')
                print(f'pval    = {pval:.5f}\n')
        chi2_df = chi2_df.append({
            'chk_col': chk_col, 
            'vs_col': vs_col, 
            'p_val': pval}
        , ignore_index=True)
        chi2_df.index.name='index'
    if print_sum:
        print(f'Found {hits} hits out of {atts} columns checked.')
    return chi2_df


def mean_value_by_field(df, field='persona_id'):
    '''
    mean_value_by_field(df, field='persona_id')

    returns dataframe showing mean values throughout original dataframe when
    controlled for unique values in source field
    '''
    mean_df = df.copy()
    mean_df[field] = mean_df[field].astype('object')
    mean_df = mean_df.groupby([field]).agg('mean').T
    mean_df['all'] = df[mean_df.index].mean()
    return mean_df



if __name__ == '__main__':
    path_prefix=''
    import sys
    sys.path.append(path_prefix)
    import wrangle
    data, datadict = wrangle.wrangle_data(path_prefix=path_prefix)
    data = add_target_to_df(data)
    recode2 = make_recode_df(df=data, data_dictionary=datadict, outputs = 2)
    print('recode2 shape:', recode2.shape)
    ctgy_cols = data.select_dtypes(['category']).columns
    quant_cols = data.select_dtypes([int,float,bool]).columns
    exclude_cols_chi2 = [
        'job_title',
        'num_employees',
        'num_researchers',
        'primary_industry',
        'types_res_used',
        'future_res',
        'job_id',
        'learning_workshop',
        'learning_conference',
        'likely_conference',
        'likely_workshop'
    ]
    list_cols = ctgy_cols.to_list() + quant_cols.to_list()
    list_cols = [col for col in list_cols if col not in exclude_cols_chi2]
    chi2_cols = [col for col in recode2.columns.to_list() if col in list_cols]
    chi2_2df = get_chi2_for_cols(recode2, chi2_cols, 'target', print_sum=True)
    print(chi2_2df[chi2_2df.p_val<=5e-02])
    persona_recode2 = mean_value_by_field(recode2)
    print(persona_recode2)



        