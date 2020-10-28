from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#from transformers import AutoTokenizer
#nltk.download('stopwords') if stop words not downloaded already
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
import numpy as np
import glob
import re
import os
import io
import heapq
import gensim
import spacy

# enddocnore = re.compile(r"<docno> ?(.*) ?<\docno>")
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
                data.append(text.read().lower()) # this converts data to lower case and lemmatizes it
        else:
            with io.open(file,'r',encoding = "ISO-8859-1") as f:
                data.append(f.read().lower()) # lemmatize answers later.
    return data


"""
    parseQuestions(data) returns a dictionary of questions as keys and their numbers as values
    Input : data -> the string that contains the questions
    Outout : question_dict -> dictionary of string key and integer value
"""
def parseQuestions(data):
    lines = data.splitlines()
    question_dict = {}
    for i in range(0, len(lines)-1, 3):
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
    for i in range(0, len(lines)):
        temp = lines[i].split()
        id_dict[temp[0]] = temp[1]
    return id_dict


def getXmlDict(topdocs):
    dict = {}
    for topdoc in topdocs:
        for k,v in genXMLListFromTopDocs(topdoc).items():
            dict[k] = sent_tokenize(v)
    return dict


"""
    genXMLListFromTopDocs(doclist) returns a dictionary of docnos as keys and the text associated with them as the value
    Input : doclist -> the string corresponding to topdoc.x, where x is the question
    Output : docdict -> dictionary of key docno and value docno's relevant text
"""
def genXMLListFromTopDocs(doclist):
    docnos = []
    texts = []
    tokens = re.split(r"[ \t\n]", doclist)
    REPLACE_DOCNO = re.compile(r"</docno>")
    REPLACE_TAG = re.compile(r"<[\/a-z\.]+>")
    buf = ""
    fillbuf = False
    for i in range(len(tokens)):
        if tokens[i] == "<docno>":
            docnos.append(tokens[min(i+1, len(tokens)-1)])
            fillbuf = True

        elif len(tokens[i]) > 7 and tokens[i][:7] == "<docno>":
            docnos.append(REPLACE_DOCNO.sub("",tokens[i][7:]))
            fillbuf = True

        if fillbuf:
            buf += REPLACE_TAG.sub("", tokens[i]) + " "
            # buf += tokens[i]+" "

        if len(tokens[i]) > 1 and tokens[i][0] == "<" and tokens[i] == "</doc>":
            texts.append(" ".join(word_tokenize(buf))) # word_tokenize adds a space between punctuation and word, formats stuff nicely
            buf = ""
            fillbuf = False

    return dict(zip(docnos, texts))


"""
    parseTopDocs(data) returns a dictionary of questions numbers as keys and the topdoc id as the value
    Input : data -> the string that maps question number to topdoc id
    Output : xml_dict -> dictionary of key TopDoc integer id number and value string xml doc
"""
def parseTopDocs(data):

    # xml_dict = xmltodict.parse(data[0])
    # print(xml_dict)
    return


"""
    corpusCounts(corpus) returns the counts of each token in every chunk of the corpus for type = 0, for type = 1 returns tf idf feature matrix
    Input: corpus -> List of chunks (sentences) from the corpus
           type   -> determines whether to do CountVectorizer or TfidfVectorizer
    Output: X -> 2d numpy array of counts for each token in a chunk if type = 0 else tf idf feature matrix
"""
def corpusCounts(corpus, type = 0, second_type = 1, vocab = {}):
    if second_type == 0:
        vectorizer = CountVectorizer() if type == 0 else TfidfVectorizer()
    else:
        vectorizer = CountVectorizer(vocabulary = vocab) if type == 0 else TfidfVectorizer(vocabulary = vocab)
    X = vectorizer.fit_transform(corpus)
    #print(vectorizer.get_feature_names())
    return X.toarray() #change back to just X for sparse matrix, this converts is to numpy array


"""
    normalizedWordFrequency(corpus, X) returns the normalizedWordFrequency of each token for each chunk : count/ size of sentence as X, and binary counts for each token in a chunk as Y
    Input: corpus -> List of chunks (sentences) from the corpus, X [This sentence is the one after removing stopwords]
    Output: X -> 2d array with normalizedWordFrequency for each token in a chunk (feature matrix)
            Y -> 2d array of binary counts for each token in a chunk. (feature matrix)
"""
def normalizedWordFrequencyMatrix(corpus, X):
    X = X.astype('float64')
    Y = np.zeros((len(X),len(X[0])))
    for i in range(len(X)):
        c = float(len(word_tokenize(corpus[i]))) # To avoid integer division
        for j in range(len(X[0])):
            X[i][j] = float(X[i][j]) / c # To avoid integer division
            Y[i][j] = 1 if X[i][j] > 0 else 0
    return X,Y


"""
    removeStopWords(sentence) removes the words that do not carry much semantic meaning and returns a list of words that are not stop words
    Input: sentence -> string to remove stop words from
    Output: filtered_sentence -> list with only relevant tokens from sentence
"""
def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    return " ".join([w for w in word_tokens if not w in stop_words])



"""
    cosineSimilarity(X, Y) returns the cosine similarity of two vectors X and Y
    Input: Vector X -> which is the question feature vector and Vector Y which is the answer feature vecotr
    Output: The cosine similarity of vectors X and Y returned as 2d array
"""
def cosineSimilarity(X, Y):
    X = X.reshape(1,-1)
    Y = Y.reshape(1,-1)
    return cosine_similarity(X, Y)[0][0] # note this is a 2d array, but will have only one value


def getTopSimilar(question_dict, id_dict, topdoc_data, stopw_lemmatize = 1, fm_type = 0, nwf_or_bow = 0):
    similarity_dict = {}
    question_list, answer_section = getQnA(question_dict, topdoc_data, id_dict, stopw_lemmatize)
    vocab = buildVocab(question_list)

    if fm_type == 0:
        V = corpusCounts(question_list, 0, 1, vocab) # gets counts
        V,W = normalizedWordFrequencyMatrix(question_list, V)
    elif fm_type == 1:
        V = corpusCounts(question_list, 1, 1, vocab) #gets tfidf matrix
    else:
        return
        # V = word2Vec

    result = [[]]
    for i in range(len(question_list)):
        if fm_type == 0:
            X = corpusCounts(answer_section[i], 0, 1, vocab)
            X,Y = normalizedWordFrequencyMatrix(answer_section[i], X)
        elif fm_type == 1:
            X = corpusCounts(answer_section[i], 1, 1, vocab)
        #print("ans sec", len(answer_section[i]), answer_section[i][0])
        for j in range(len(answer_section[i])):
            s = question_list[i] + "," + answer_section[i][j]
            #print("V", V.shape)
            similarity_dict[s] = cosineSimilarity( V[i], X[j]) if fm_type != 2 else getEuclideanDistance(V[i], X[j])
        result[i].append(heapq.nlargest(10, similarity_dict.keys(), key = similarity_dict.get)) #Priority queue for top 10 answers, send this to write to file
    return result


"""
    buildVocab(question_list) goes through all questions to build vocabulary out of questions, this will be used to make the feature matrix
    Input: question_list -> list of questions(strings)
    Output: vocab -> a dictionary of all unique question tokens
"""
def buildVocab(question_list):
    vocab = {} # Using dict instead of set because CountVectorizer only takes dictionary as parameter
    idx = 0
    for question in question_list:
        for token in word_tokenize(question):
            if token not in vocab:
                vocab[token] = idx
                idx = idx + 1
    return vocab


"""
    writeToFile(question_list, question_dict, answer_list, file_name) creates new file with the prediction results
    Input: question_list -> The list of all questions
           question_dict -> Dictionary that Maps each question to its number
           answer_list -> List of lists where each list_i inside the list refers to the answers to question_i
           file_name -> This specifies what the prediciton file should be called
    Output: Creates a file with the answer patterns
"""
def writeToFile(question_list, question_dict, answer_list, file_name): #answer_list is a list of lists
    file = open(file_name, "w")
    for i in range(len(question_list)):
        file.write("qid "+id_dict.get(question_list[i])+"\n")
        for answer in answer_list[i]:
            file.write(answer+"\n")


"""
    lemmatize(sent_chunk) takes in a sentence, tokenizes it and then lemmatize each word and returns the sentence with all words lemmatized
    Input: setn_chunk -> a sentence or a string chunk with words that are not lemmatized
    Output: a sentence or a string chunk with words that are lemmatized
"""
def lemmatize(sent_chunk):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(wd, get_word_net(wd)) for wd in word_tokenize(sent_chunk)])


def addPosTags(text):
    text = word_tokenize(text)
    text_with_tags = pos_tag(text) # List of tuples, access POS tag
    return text_with_tags


def getWindowList(text, type):
    return


def get_word_net(word):
# Citation : https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def getBertData(question, answer):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoded_input = tokenizer(question, answer)
    return
    # tokenizer.decode(encoded_input["input_ids"]) to decode features, for debugging


def getEuclideanDistance(X, Y):
    X = X.reshape(1,-1)
    Y = Y.reshape(1,-1)
    return euclidean_distances(X,Y)[0][0]


def getQnA(question_dict, topdoc_data, id_dict, stopw_lemmatize, ):
    answer_section = [[]]*len(question_dict)
    id = 0
    question_list = []
    for k,v in question_dict.items():
        if stopw_lemmatize == 1:
            question_list.append(lemmatize(removeStopWords(k)))
        else:
            question_list.append(k)
        for sent_chunk in topdoc_data[id_dict[v]]:
            if stopw_lemmatize == 1:
                answer_section[id].append(lemmatize(removeStopWords(sent_chunk)))
            else:
                answer_section[id].append(sent_chunk)
        id = id + 1
    return question_list, answer_section


data = getData("training/qadata/", 0)
question_dict = parseQuestions(data[0])
id_dict = parseRelevantDocs(data[1])
# corpus = [
# 'This is the first document.',
# 'This document is the second document.',
# 'And this is the third one.',
# 'Is this the first document?']
topdoc_data = getData("training/topdocs/", 1)
# addPosTags("suvinay bothra ate breakfast at twelve PM , kartikey played a game on october seventeenth eats eated fast fasted ; s 123 what's ")
# lemmatize("walked walk walks hear heard hears serve served service go went gone")
xml_dict = getXmlDict(topdoc_data)
results = getTopSimilar(question_dict, id_dict, xml_dict)



# print(topdoc_data[0])
# parseTopDocs(topdoc_data)
