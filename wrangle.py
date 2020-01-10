import pandas as pd
import numpy as np
import json

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
    id_col = 'q01b'
    experience_col = 'research_years'

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
    research_columns = ['q02a', 'q02b', 'q02c', 'q02d', 'q02e',
                        'q02f', 'q02g', 'q02h', 'q02i', 'q02j',]
        
    max_research = df[research_columns].apply(max, axis = 1)
    return max_research
    


encoder_dictionary = {
    'important5': {
        'Not at all Important' : 0,
        'Somewhat Important' : 1,
        'Neutral' : 2,
        'Important' : 3,
        'Very Important': 4,
        np.NaN: None
    },
    'likely5': {
        'Extremely unlikely' : 0,
        'Unlikely' : 1,
        'Neutral' : 2,
        'Likely' : 3,
        'Extremely likely': 4,
        np.NaN: None
    },
    'often4': {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Often': 3,
        np.NaN: None
    },
    'often5': {
        'Never': 0,
        'Occasionally': 1,
        'Neutral': 2,
        'Often': 3,
        'All or almost all': 4,
        np.NaN: None
    },
    'tenure6': {
        "0-1 year": 0,
        "1-3 years": 1,
        "3-5 years": 2,
        "5-7 years": 3,
        "7-10 years": 4,
        "10+ years": 5,
        np.NaN: None
    },
    'q03dict': {
        "1-25 employees": 0,
        "26-100 employees": 1, 
        "101-500 employees": 2,
        "501-1,000 employees": 3,
        "1,000+ employees": 4,
        np.NaN: None
    },
    'q04dict': {
        "1-5 employees": 0,
        "6-10 employees": 1,
        "11-50 employees": 2,
        "50+ employees": 3,
        np.NaN: None
    }
}


map_df = pd.read_csv('../kev/question_library.txt').dropna().set_index('qid2')
map_df['col'] = map_df.index



def encode_columns(df, cols=map_df.col, encoder=encoder_dictionary, check_dicts=map_df.cat_lib):
    '''

    '''
    for col in cols:
        use_dict = encoder[check_dicts[col]]
        encoded = df[col].apply(lambda x: use_dict[x])
        df[col] = encoded
    return df

  
def wrangle_data():
    #get data
    data = pd.read_excel('../kev/survey_responses.xlsx', )

    #encode columns
    encode_columns(data)

    #find max research years to be used in label category
    research_years = get_max_research_years(data)

    #apply labels
    data['persona_id'] = data.join(pd.DataFrame(research_years, columns=['research_years'])).apply(get_labels, axis=1)

    return data
