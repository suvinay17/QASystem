import gensim
import numpy as np
from nltk import word_tokenize, sent_tokenize

class WTV:
    """
        init(corpus) creates new Word2Vec model trained on the provided list of questions and answers
        Input: questions -> list of questions as strings
        answers -> list of answers as strings
        Output: A trained W2V model which can be used to provide sentences feature vectors
    """
    # scope for improvement:
    # - could pre-train the model and store the keyvectors to disk? Or keep it so that it can be trained on the fly for unknown words? Or keep it pretrained and then
    # use it for on the fly training if necessary too?
    # - maybe instead of adding np.zeros add feature vector for unknown word?
    def __init__(self, questions, texts, minCount = 1):
        corpus = questions + texts
        tokens = [word_tokenize(sent) for sent in corpus]
        self.model = gensim.models.Word2Vec(tokens, min_count = minCount, window = 5) # train a word2vec model which ignores words occurring less than 1 times, with window size 5.
        self.wv = self.model.wv # dictionary mapping words to their w2v numpy vectors

    '''
    wtv.getAvgWordVec() creates the sentence feature vector by averaging the word vectors for all the words in the sentence.
    input: sentence -> the untokenized sentence to get feature vector from.
    output: the feature vector for the sentence.
    '''
    def getAvgWordVec(self, sentence):
        words = word_tokenize(sentence)
        sumv = np.zeros(100)
        for w in words:
            sumv += self.wv[w] if w in self.wv else np.zeros(100) # if word in vocab add its feature vector otherwise add 0s

        return sumv/len(words)

    '''
    wtv.getAvgWordVecFromList() takes a list of untokenized sentences (e.g. ["Hello world.", "I am Jacob.", "How is it going?"]) as input in case it's already available somewhere.
    '''
    def getAvgWordVecFromList(self, sentences):
        out = []
        for s in sentences:
            out.append(self.getAvgWordVec(s))

        return out
