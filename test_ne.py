import nltk
import gensim
from gensim.models import Word2Vec
from dateparser.search import search_dates


sentence = "Suvinay Bothra drank a cup of coffee on 06/06/2020 at Starbucks. He's a sick bastard. I'm trying to solve a Deep Learning text classification problem, so I have to vectorize the text input with Word2Vec to feed it into a neural network."

sents = nltk.sent_tokenize(sentence)
tokens = [nltk.word_tokenize(s) for s in sents]

# print(tokens, "\n----------")
tokens_tagged = nltk.pos_tag(tokens[0])

net = nltk.ne_chunk(tokens_tagged)

for chunk in net:
    if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))

print(search_dates(sentence))
