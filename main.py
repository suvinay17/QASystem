from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords') if stop words not downloaded already
import numpy as np
import glob
import re
import os
import xmltodict
import io


"""
    getData(path) returns a list of strings where each string is .txt or xml file in a given path
    Input : path -> the directory from where the .txt files are read
            type -> 0 if we are parsing .txt 1 if we are parsing xml
    Output : data -> list of strings of a txt or xml depending on type argumnet
"""
def getData(path, type):
    files = glob.glob(os.path.join(os.getcwd(), path, "*"))
    data = []
    for file in files:
        if type == 0:
            with open(file) as text:
                data.append(text.read().lower()) # this converts data to lower case
        else:
            with io.open(file,'r',encoding = "ISO-8859-1") as f:
                data.append(f.read().lower())
    return data

"""
    parseQuestions(data) returns a dictionary of questions as keys and their numbers as values
    Input : data -> the string that contains the questions
    Outout : question_dict -> dictionary of string key and integer value
"""
def parseQuestions(data):
    lines = data.splitlines()
    question_dict = {}
    for i in range(0, len(lines), 3):
        temp = lines[i].split()
        question_dict[lines[i+1]] = temp[1]
    return question_dict


"""
    parseRelevantDocs(data) returns a dictionary of questions numbers as keys and the topdoc id as the value
    Input : data -> the string that maps question number to topdoc id
    Outout : id_dict -> dictionary of key integer question number and value string relevant doc id
"""
def parseRelevantDocs(data):
    lines = data.splitlines()
    id_dict = {}
    for i in range(0, len(lines)-1):
        temp = lines[i].split()
        #print(temp)
        id_dict[temp[0]] = temp[1]
    return id_dict


"""
    parseTopDocs(data) returns a dictionary of questions numbers as keys and the topdoc id as the value
    Input : data -> the string that maps question number to topdoc id
    Outout : xml_dict -> dictionary of key TopDoc integer id number and value string xml doc
"""
def parseTopDocs(data):
    # xml_dict = xmltodict.parse(data[0])
    # print(xml_dict)
    return


"""
    cosineSimilarity(X, Y) returns the cosine similarity of two vectors X and Y
    Input: Vector X which is the question feature vector and Vector Y which is the answer feature vecotr
    Output: The cosine similarity of vectors X and Y returned as 2d array
"""
def cosineSimilarity(X, Y):
    X.reshape(1,-1)
    Y.reshape(1,-1)
    return cosine_similarity(X, Y) # note this is a 2d array, but will have only one value


"""
    corpusCounts(corpus) returns the counts of each token in every chunk of the corpus.
    Input: corpus: List of chunks (sentences) from the corpus
    Output: X: 2d array of counts for each token in a chunk
"""
def corpusCounts(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    #print(vectorizer.get_feature_names())
    return X.toarray() #change back to just X for sparse matrix


"""
    corpusCounts(corpus) returns the normalizedWordFrequency of each token for each chunk : count/ size of sentence as X, and binary counts for each token in a chunk as Y
    Input: corpus: List of chunks (sentences) from the corpus, X
    Output: X: 2d array with normalizedWordFrequency for each token in a chunk (feature matrix)
            Y: 2d array of binary counts for each token in a chunk. (feature matrix)
"""
def normalizedWordFrequencyMatrix(corpus, X):
    Y = [[]] #Might Change X, Y to Numpy Arrays
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = X[i][j]/len(corpus[i])
            Y[i][j] = 1 if x[i][j] > 0 else 0
    return X,Y


"""
    removeStopWords(sentence) removes the words that do not carry much semantic meaning and returns a list of words that are not stop words
    Input: sentence: string to remove stop words from
    Output: filtered_sentence: list with only relevant tokens from sentence
"""
def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


data = getData("training/qadata/", 0)
question_dict = parseQuestions(data[0])
# print(data[0])
id_dict = parseRelevantDocs(data[1])
# print(id_dict)
topdoc_data = getData("training/topdocs/", 1)
# print(topdoc_data[0])
# parseTopDocs(topdoc_data)
