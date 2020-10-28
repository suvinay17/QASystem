import spacy

'''NERecognizer class implements an Named Entity Recognizer which can output relevant entities based on question type.'''
class NERecognizer():
    # initializes the model
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

    """
        ner.getEntities(sent) returns a dictionary mapping strings to a list of strings where each key is a possible entity label (PERSON, GPE, DATE, TIME...)
        and the value is the list of all entities in the sentence matching that label
        Input : sent -> the sentence to get entities from (must be a whole string: e.g: "Hi it is 3 PM", not ['Hi', 'it', 'is', '3', 'pm'])
        Output : entities -> dictionary mapping labels to a list of words matching that entity.
    """
    def getEntities(self, sent):
        doc = self.model(sent)
        entities = {}
        for e in doc.ents:
            if e.label_ not in entities:
                entities[e.label_] = [e.text]
            else:
                entities[e.label_].append(e.text)

        return entities

    """
        ner.getAnsCandidates(questionType, answer) returns a list of words from the answer matching question type (can be "who", "when", "where"), otherwise returns []
        Input : questionType -> the type of question ("who", "when", "where")
                anser -> The answer string
        Output : ans_cands -> list of entities from sentence matching the question type (PERSON for who, DATE and TIME for when, and GPE and LOC for where)
    """
    def getAnsCandidates(self, questionType, answer):
        entities = self.getEntities(answer)
        ans_cands = []

        if questionType == "who":
            ans_cands = entities['PERSON'] if 'PERSON' in entities else []

        elif questionType == "when":
            ans_cands = entities['DATE'] if 'DATE' in entities else []
            ans_cands += entities['TIME'] if 'TIME' in entities else []


        elif questionType == "where":
            ans_cands = entities['GPE'] if 'GPE' in entities else []
            ans_cands += entities['LOC'] if 'LOC' in entities else []

        return ans_cands
