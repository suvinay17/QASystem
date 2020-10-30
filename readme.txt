One must have pip installed to download all the libraries needed to run this class:

Then do the following:

pip install scikit-learn
pip install nltk
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy validate

Then, to use stopwords, lemmatizer and pos tagging, the following must be downloaded,
run this on your python interpreter:

nltk.download('stopwords') if stop words not downloaded already
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

After this to run the code:
python3 main.py
This generates the prediction files for test and training

Uncomment line 345 to 355 to also generate test predictions

PLEASE NOTE : The train prediction file is called prediction.txt where as test prediction file is called test_prediction.txt
For using inferior #Extraction Strategy 2 as explained  in the report when the WriteToFile() method is invoked on lines 339 and 354, the method NER.getAnsFromQuestionListWithContext() 
should be used instead of NER.getAnsFromQuestionList.
