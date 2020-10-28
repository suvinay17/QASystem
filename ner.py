import spacy
import nltk

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
                answer -> The answer string
        Output : ans_cands -> list of entities from sentence matching the question type (PERSON for who, DATE and TIME for when, and GPE and LOC for where)
    """
    def getAnsCandidates(self, questionType, answer):
        entities = self.getEntities(answer)
        ans_cands = []

        if questionType == "who":
            ans_cands =  entities['PERSON'] if 'PERSON' in entities else []

        elif questionType == "when":
            ans_cands =  entities['DATE'] if 'DATE' in entities else []
            ans_cands += entities['TIME'] if 'TIME' in entities else []


        elif questionType == "where":
            ans_cands =  entities['GPE'] if 'GPE' in entities else []
            ans_cands += entities['LOC'] if 'LOC' in entities else []
            ans_cands += entities['FAC'] if 'FAC' in entities else []

        elif questionType == "what":
            ans_cands = []

        elif questionType == "how":
            ans_cands = []

        return ans_cands

    def getAnsFromQuestionList(self, questions, answer_section):
        question_types = ["who", "when", "where", "what", "how"]
        candidate_answers = [[] for i in range(len(questions))]
        for i in range(len(questions)):
            # determine question type
            question_words = nltk.word_tokenize(questions[i])
            q_type = None
            for type in question_types:
                if type in question_words:
                    q_type = type

            for j in range(len(answer_section[i])):
                entities = self.getAnsCandidates(q_type, answer_section[i][j])

                if q_type == "what" or q_type == "how":
                    # in these cases, just return whatever noun phrases you can find in the sentence.
                    tagged_ans = nltk.pos_tag(nltk.word_tokenize(answer_section[i][j]))

                    w = 0
                    noun_phrases = []
                    buffer = ""
                    rec_flag = False # flag to start recording
                    while w < len(tagged_ans):
                        if tagged_ans[w][1] in ["NN", "NNP", "NNS"]:
                            if rec_flag == False:
                                buffer += tagged_ans[w][0]
                                rec_flag = True

                            else:
                                buffer += " " + tagged_ans[w][0]
                        else:
                            if buffer != "":
                                noun_phrases.append(buffer)
                            buffer = ""
                            rec_flag = False

                        w += 1

                    candidate_answers[i].append(noun_phrases)

                else:
                    if len(entities) != 0:
                        candidate_answers[i].append(entities)

        return candidate_answers
