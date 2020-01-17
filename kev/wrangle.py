import pandas as pd
import numpy as np
import json


# Declare rules for encoding
encoder_dictionary = {
    "important5": {
        "Not at all Important" : 0,
        "Somewhat Important" : 1,
        "Neutral" : 2,
        "Important" : 3,
        "Very Important": 4,
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
    "often4": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        np.NaN: None
    },
    "often5": {
        "Never": 0,
        "Occasionally": 1,
        "Neutral": 2,
        "Often": 3,
        "All or almost all": 4,
        np.NaN: None
    },
    "q03_cats": {
        "1-25 employees": 0,
        "26-100 employees": 1, 
        "101-500 employees": 2,
        "501-1,000 employees": 3,
        "1,000+ employees": 4,
        np.NaN: None
    },
    "q04_cats": {
        "1-5 employees": 0,
        "6-10 employees": 1,
        "11-50 employees": 2,
        "50+ employees": 3,
        np.NaN: None
    },
    "q09a_cats": {
        "Yes, I was taught how to conduct research ": 1,
        "No, I was not taught how to conduct research": 0,
    },
    "q17a_cats": {
        "Retreat/workshop: < 50 attendees": 1,
        "Small conference: < 300 attendees ": 2,
        "Mid-size conference : 300-500 attendees": 3,
        "Large conference: 500+ attendees": 4,
        "No preference": 6,
        "It Depends": 7,
        "Virtual": 8,
    },
    "q18_cats": {
        "Single-Track": 1,
        "Multi-Track": 2,
        "Unconference": 3,
        "Mixed-Type": 4,
        "No Preference": 5,
    },
    "tenure6": {
        "0-1 year": 0,
        "1-3 years": 1,
        "3-5 years": 2,
        "5-7 years": 3,
        "7-10 years": 4,
        "10+ years": 5,
        np.NaN: None
    },
    "yesno10": {
        "Yes": 1,
        "No": 0,
    }

}


# Declare rules for joining manually reviewed columns
join_files = {
    'q09a': 'data_files/q09_to_categories.txt',
    'q17a': 'data_files/q17_to_categories.txt',
    'q18a': 'data_files/q18_to_categories.txt',
}


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
        if x[experience_col] in [5]:
            return 3
        if x[experience_col] in [2,3,4]:
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
    research_columns = ['q08a', 'q08b','q08c', 'q08d', 'q08e', 
                        'q08f', 'q08g', 'q08h', 'q08i', 'q08j']
        
    max_research = df[research_columns].apply(max, axis = 1)
    return max_research
    

# Get data dictionary
def get_data_dictionary(path_prefix='', dictionary_path='data_files/data_dictionary.xlsx'):
    data_dictionary = pd.read_excel(path_prefix + dictionary_path).set_index('qid').sort_index()
    return data_dictionary


# Get mapping dictionary
def get_mapping_dictionary(data_dictionary):
    map_df = data_dictionary[['encoder']].dropna()
    map_df['col'] = map_df.index
    return map_df


def encode_columns(df, map_df, encoder=encoder_dictionary):
    '''

    '''
    allcols=map_df.col
    check_dicts=map_df.encoder
    cols=[chkcol for chkcol in allcols if chkcol in df.columns]
    for col in cols:
        use_dict = encoder[check_dicts[col]]
        encoded = df[col].apply(lambda x: use_dict[x])
        df[col] = encoded
    return df


def append_join_files(df, reqd_cols, path_prefix='', join_files=join_files):
    chk_cols = [col for col in reqd_cols if col in df.columns]
    for key in join_files:
        file_path = join_files[key]
        join_df = pd.read_csv(path_prefix + file_path).set_index('q00')
        df = df.join(join_df)
        chk_cols = [col for col in reqd_cols if col in df.columns]
    return df


def impute_q09(df):
    re_adds = (df[[
        'q09a',
        'q09b',
        'q10',
    ]][
        (df.q09a.isna()) & 
        (df.q10.isna()==False)
    ])
    re_add_idxs = re_adds.index.tolist()
    for idx in re_add_idxs:
        df.loc[idx, 'q09a'] = 'Imputed Yes'
        df.loc[idx, 'q09b'] = 1
    return df
    

def drop_reqd_nas(df, reqd_cols):
    chk_cols = [col for col in reqd_cols if col in df.columns]
    na_df = df[chk_cols].copy()
    na_df['nulls'] = na_df.isna().sum(axis=1)
    df = df.join(na_df[['nulls']])
    df=df[df.nulls==0].drop(columns=['nulls'])
    
    return df


def reset_column_types(df, data_dictionary):
    chk_cols = data_dictionary[['column_name', 'data_type']].set_index('column_name')
    set_dict = chk_cols.to_dict()['data_type']
    pops = []
    for key in set_dict:
        if key not in df.columns:
            pops.append(key)
    for pop in pops:
        set_dict.pop(pop)
    df = df.astype(set_dict)
    return df


def wrangle_data(
    path_prefix='../', 
    data_path = 'data_files/survey_responses.xlsx',
    dictionary_path = 'data_files/data_dictionary.xlsx',
    encoder=encoder_dictionary,
    join_files=join_files
):
    '''
    reads data from survey_responses.xlsx and applies the following transformations:
        - encodes categoriacal variables
        - calculates and applies persona labels
        - assigns more descritive names for columns from data_dictionary

    args: None
    returns: pandas DataFrame
    '''
    # Get data
    data = pd.read_excel(path_prefix + data_path)
    data['idx'] = data.q00
    data.set_index('idx', inplace = True)

    # Get data dictionary
    data_dictionary = get_data_dictionary(path_prefix, dictionary_path=dictionary_path)
    
    #Set encoder map
    map_df = get_mapping_dictionary(data_dictionary=data_dictionary)
    
    #encode columns
    encode_columns(data, map_df=map_df, encoder=encoder,)

    
    # Append join columns
    reqd_cols = data_dictionary.index[(data_dictionary.is_required)]
    reqd_cols = reqd_cols[reqd_cols != data.index.name]
    data = append_join_files(data, reqd_cols=reqd_cols, path_prefix=path_prefix, join_files=join_files)
    
    # Impute missing q09 values from q10
    data = impute_q09(data)
    
    # Drop required NAs
    data = drop_reqd_nas(data, reqd_cols=reqd_cols)
    
    #find max research years to be used in label category
    research_years = get_max_research_years(data)

    # Apply labels
    data['persona_id'] = (
        data
        .join(pd.DataFrame(research_years, columns=['research_years']))
        .apply(get_labels, axis=1)
        )

    # Update column names from data dictionary
    data.rename(columns = data_dictionary.column_name, inplace = True)
    data = data[data_dictionary.column_name]

    #set index
    data.set_index('resp_id', inplace = True)

    # Reset Column Types
    data = reset_column_types(data, data_dictionary=data_dictionary)
    
    return data, data_dictionary
