import pandas as pd
import numpy as np

def get_labels(x):
    '''
    assigns subjects to persona groups based on their job title and years of reaearch

    use with df.apply(get_labels, axis = 1) and make sure the 'id_col' is set to what ever the 'Job Taxo ID' column is names and the 'max_experience' is derrived from other functions

    args: x - pandas datafram row
    returns: pandas series of persona ids
    '''
    #taxo ids we assume indicate personas
    exec_and_consult_ids = [7,8]
    specialist_ids = [3,4,5,9]
    other_ids = [1,2,6]

    #open variable for calling id column, coded this way for easy change should we rename the columns
    id_col = 'Job Taxo ID'
    experience_col = 'max_experience'

    #assign persona classifiers based on their taxo id first followed by their max years of experience
    if x[id_col] in exec_and_consult_ids:
        return 1
    if x[id_col] in specialist_ids:
        return 2
    if x[id_col] in other_ids:
        if x[experience_col] in [4,5]:
            return 3
        if x[experience_col] in [2,3]:
            return 4
        if x[experience_col] in [0,1]:
            return 5
    ### return a null if no conditions are met
    return np.nan

def get_max_research_years(df):
    '''
    function takes in dataframe, grabs all columns pertaining to research years (q2:a-j) encodes them and labels subject the value from the column with the higest value

    args: df - pandas dataframe
    returns: max_research - pandas series
    '''
    #define columns that describe subject's research experience
    research_columns = ['Conducting Research.1',
                        'Analyzing Research.1',
                        'Buying Research Reports.1',
                        'Managing Research Projects.1',
                        'Observing Research.1',
                        'Planning Research.1',
                        'Teaching Research.1',
                        'Advocating for Research.1',
                        'Hiring Research Vendors.1',
                        'Leading a Research Team or Organization.1']
    
    #encode those columns
    encoded_research_columns = pd.DataFrame()
    for i in research_columns:
        col = df[i].apply(encode_years)
        encoded_research_columns['encoded ' + i] = col
    max_research = encoded_research_columns.apply(max, axis = 1)
    return max_research
    
    
    
def encode_years(x):
    '''
    use pd.series.apply(encode_years) to encode column
    this function is used in get_max_research_years

    args: x: string
    returns: int
    '''
    if '0-1' in str(x):
        return 0
    elif '1-3' in str(x):
        return 1
    elif '3-5' in str(x):
        return 2
    elif '5-7' in str(x):
        return 3
    elif '7-10' in str(x):
        return 4
    elif '10+' in str(x):
        return 5
    else: 
        return x

    