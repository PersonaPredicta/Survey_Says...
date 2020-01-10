#Ah la la, Ah la la, Gimme Three wishes, I wanna be that Dirtyfinger and his six...

from textblob import TextBlob
import pandas as pd
from itertools import islice

#Each function will be applied to the columns with the Series.apply() method 

def find_polarity(input_text):
    """
    Takes one document and outputs a polarity score. -1 is Very Negative, 1 is Super Positive
    This function should be applied to a column and outputted to a new column.
    """
    return TextBlob(input_text).sentiment.polarity

def find_subjectivity(input_text):
    """
    Takes one document and outputs a subjectivity score. 0 is cold objectivity, 1 is very subjective
    This function should be applied to a column and outputted to a new column.
    """
    return TextBlob(input_text).sentiment.subjectivity




# def create_sentiment_scores(orig_text):
#     """
#     Accepts a Series/df_column that of text responses. Outputs the polarity, subjectivity, and other scores 
#     in a new dataframe.
#     """

#     pass

#I'm just gonna write a few two line functions that append to a blank DF
    # for index, row in islice(orig_df.iterrows(), 0, None):

    #     new_entry = []
    #     text_lower = to_lower(row['Responses'])
    #     blob = TextBlob(text_lower)
    #     sentiment = blob.sentiment
            
    #     polarity = sentiment.polarity
    #     subjectivity = sentiment.subjectivity
            
    #     new_entry += [row['Response Date'],text_lower,sentiment,subjectivity,polarity]
            
    #     single_survey_sentimet_df = pd.DataFrame([new_entry], columns=COLS)
    #     df = df.append(single_survey_sentimet_df, ignore_index=True)