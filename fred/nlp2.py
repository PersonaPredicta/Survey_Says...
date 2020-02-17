from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob
import nltk

import pyLDAvis
import pyLDAvis.sklearn

import numpy as np
import pandas as pd

import unicodedata
import re


#Prepare

def create_biganswer_col(df):
    """
    Accepts a DataFrame as an input. The dataframe will be reduced to the columns that are of a string/object.
    Nulls will be made into a string value called 'n/a'. Each column is double-checked as a string type
    by using the .astype() method.
    The returned value is a Series that is all the column's values smashed into one column.
    """
    df_quals = df.select_dtypes('object')
    df_quals = df_quals.fillna('n/a')
    df_quals = df_quals.astype('str')
    col_names = list(df_quals.columns)
    list_of_cols = [df_quals[i] for i in col_names]
    big_answer = list_of_cols[13]
    for i in range(len(list_of_cols)-1):
        big_answer = big_answer + list_of_cols[i]
    return big_answer

#I would appened this big answer column back to the dataframe you are working with.
#Maybe just the df.select_dtypes('object'). That would be just the qualitative answers.

def lemmatize(text):
    """
    accept some text and return the text after applying lemmatization to each word.
    Use with .apply to a Pandas Series
    """
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    return ' '.join(lemmas)

def basic_clean(text):
    """
    Lowercase everything
    Normalize unicode characters
    Replace anything that is not a letter, number, whitespace or a single quote
    """
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    text = re.sub(r"[\r|\n|\r\n]+", ' ', text)
    return text


def set_stop_words(stop_words):
    """
    Takes a list of strings. These strings will be appended to the 'english' stop words used in sklearn
    objects that have a stopword parameter.
    Returns a very big list. Assign this to the stopwords parameter in the sklearn objects.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
    stopWords = stop_words + list(esw) 
    return stopWords

def create_wordcount_matrix(input_column, max_df=0.3, min_df=2, ngram=(1,3), stop_words='english'):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a wordcount for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, count_vect

def create_LDA_model(matrix, vector, n_topics=5):
    """
    Requires the matrix and vector created by the create_wordcount_matrix function.
    Input for the pyLDAvis exploration.
    """
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    return LDA


#Wordcounts or Ngrams Counts

def find_word_counts(input_column, max_df=.3, min_df=2, ngram_range=(1,3), stop_words='english'):
    """
    Accepts a column of text. Uses default parameters for the WordCount Vectorizer.
    Returns a dataframe with the word list, and their frequency as the other column.
    """
    input_column = input_column.dropna().apply(basic_clean)
    input_column = input_column.apply(lemmatize)
    cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)   
    blob = cv.fit_transform(input_column)    
    word_list = cv.get_feature_names()    
    count_list = blob.toarray().sum(axis=0)
    word_counts = {'word_list': word_list, 'count_list': count_list}
    df_word_count = pd.DataFrame(data=word_counts)
    return df_word_count

#Topic Modeling

def show_LDA_topic_words(matrix, vector, n_topics=5, n_words=5):
    """
    Accepts a doc_term_matrix and a fitted vectorizing model (from previous function). 
    Fits the LDA algorithm to the matrix.
    Uses the features/words from the vectorizing model.
    Prints the top n words for n topics.
    """
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    LDA.fit(matrix)
    for i, topic in enumerate(LDA.components_): 
        print(f"Top {n_words} words for topic #{i}:") 
        print([vector.get_feature_names()[i] for i in topic.argsort()[-n_words:]]) 
        print('\n')

def assign_topic_column(input_column, max_df=.8, min_df=2, stop_words='english', ngram_range=(1,3), n_components=3):
    """
    takes a pandas column/Series. Runs through the topic modeling pipeline. Assigns the topic_id
    to each row. topic_id with highest probability is assigned.
    Returns a Series that is a int value of the topic (0--n)
    """
    input_column = input_column.astype('str')
    input_column = input_column.fillna('nan')
    input_column = input_column.apply(basic_clean)
    input_column = input_column.apply(lemmatize)
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
    LDA.fit(doc_term_matrix)
    lda_H = LDA.transform(doc_term_matrix)
    topic_doc_df = pd.DataFrame(lda_H)
    return topic_doc_df.idxmax(axis=1), topic_doc_df.max(axis=1)

def assign_topic_column_probability(input_column, max_df=.8, min_df=2, stop_words='english', ngram_range=(1,3), n_components=3):
    """
    takes a pandas column/Series. Runs through the topic modeling pipeline.
    Returns a Series that is a probability score for the most likely topic.
    """
    input_column = input_column.fillna('nan')
    input_column = input_column.astype('str')
    input_column = input_column.apply(basic_clean)
    input_column = input_column.apply(lemmatize)
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words, ngram_range=ngram_range)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
    LDA.fit(doc_term_matrix)
    lda_H = LDA.transform(doc_term_matrix)
    topic_doc_df = pd.DataFrame(lda_H)
    return topic_doc_df.max(axis=1)

