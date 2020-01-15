#Ah la la, Ah la la, Gimme Three wishes, I wanna be that Dirtyfinger and his six...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob
import nltk

import pyLDAvis
import pyLDAvis.sklearn

import numpy as np
import pandas as pd


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


def create_sentiment_df(input_column):
    """
    Accepts a column of text data. Returns a dataframe with the original text with their polarity and 
    subjectivity scores appended to it.
    """
    cols = ['text','polarity','subjectivity']
    df = pd.DataFrame(columns=cols)
    df['text'] = input_column
    df['polarity'] = input_column.apply(find_polarity)
    df['subjectivity'] = input_column.apply(find_subjectivity)
    return df


def create_tfidf_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1)):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a tf-idf score for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram)
    doc_term_matrix = tfidf_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, tfidf_vect


def create_wordcount_matrix(input_column, max_df=0.8, min_df=2, ngram=(1,1)):
    """
    Creates a feature matrix. Matrix is as wide as the terms that meet the min/max parameters. Each document/row
    will have a wordcount for each term.
    Can find ngrams, but has default set to 1-word ngrams. Set ngrams to (1,n) to look for ngrams.
    """
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram)
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
    return doc_term_matrix, count_vect

#USE MATRIX AND VECTORS from last two funcitons as parameters for the next two functions.
# create_tfidf_matrix -> show_nmf_topic_words
# create_wordcount_matrix -> show_LDA_topic_words

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

def show_nmf_topic_words(matrix, vector, n_topics=5, n_words=5):
    """
    Accepts a doc_term_matrix and a fitted vectorizing model (from previous function). 
    Fits the NMF algorithm to the matrix.
    Uses the features/words from the vectorizing model.
    Prints the top n words for n topics.
    """
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(matrix )
    for i, topic in enumerate(nmf.components_): 
        print(f"Top {n_words} words for topic #{i}:") 
        print([vector.get_feature_names()[i] for i in topic.argsort()[-n_words:]]) 
        print('\n')


def find_top_documents_per_topic(lda_W, documents):
    """
    lda_H is LDA fitted to a tf/tfidf matrix.components_
    lda_W is LDA.fit(matrix).transform(matrix)
    n_top_docs is a integer for how many documents you want to represent a topic
    """
    df_W = pd.DataFrame(lda_W, index=documents.index)
    indeces = df_W.idxmax(axis=0)
    counter = 0
    for i in indeces:
        print(f'Top Document for Topic {counter}: \n')
        print(documents[i] + '\n')
        counter += 1

def show_plLDAvis_dashboard(lda_fitted_model, doc_term_matrix, vector):
    pyLDAvis.sklearn.prepare(lda_fitted_model, doc_term_matrix, vector)


def make_word_counts(input_column):
    """
    I should probably stop at the doc term matrix.
    """
#words that appear in less than 80% of the document and appear in at least 2 documents. Also removes the stop
#words that are in sklearns stopword list.
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
#The doc term matrix is a sparse matrix that has as many columns as there are words in the corpus/input_column.
#count_vect is now 'fitted' to the corpus we gave it. It has some useful methods and attributes to use later.
    doc_term_matrix = count_vect.fit_transform(input_column.values.astype('U'))
#I'm unsure about the astype('U'), but it was in the tutorial I copied this stuff from

#n_components defines how many topic/clusters will be sorted. 
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(doc_term_matrix)

#LDA, after fitting to the document term matrix, is an array of 5. Each element is a topic, with a probability of
# each word/feature/column in the doc_term_matrix    
    first_topic = LDA.components_[0]
#This is the top 5 words/features in the first topic. It returns the index, so we have to look up the actual
#word in the count_vect object
    first_topic.argsort()[-5:]
#loops through the 5 words, and passes them to the count_vect.get_feature_name method
    for i in first_topic.argsort()[-5:]: 
        print(count_vect.get_feature_names()[i])

# Loops through the 5 topics, which are the LDA.components. 
# The inner loop is is the loop above this one 
    for i, topic in enumerate(LDA.components_): 
        print(f"Top 5 words for topic #{i}:") 
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-5:]]) 
        print('\n')
#These are an ndarray that is as wide as the amount of topics defined by LDA. Each value in each vector
#is the probability for each topic.
    topic_values = LDA.transform(doc_term_matrix)
#This will select the value in each row with the highest probability. The best guess for what its topic is.
    topic_values.argmax(axis=1)
    pass
