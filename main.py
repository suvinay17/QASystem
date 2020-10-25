import numpy as np
import glob
import re
import os
import xmltodict
import io

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
        id_dict[temp[0]] = temp[1]
    return id_dict

def genXMLListFromTopDocs(doclist):
    # iterate through each line of doclist. if you see <docno>, get the id between <docno> and </docno>. save it as a variable you're
    # going to use as a key for your hashmap. keep going until you hit a <text> tag. initialize a buffer, and store everything that occurs
    # until an </text> tag. Use regex r"<.+>" to catch all xml tags in the buffer, and sub them with an empty string. Put the resultant text as hashmap val
    # to the key.
    docnos = []
    texts = []
    tokens = re.split(r"[ \t\n]", doclist)
    REPLACE_DOCNO = re.compile(r"</docno>")
    REPLACE_TAG = re.compile(r"<[\/a-z]+>")
    buf = ""
    fillbuf = False
    for i in range(len(tokens)):
        if tokens[i] == "<docno>":
            docnos.append(tokens[min(i+1, len(tokens)-1)])
            fillbuf = True

        elif len(tokens[i]) > 7 and tokens[i][:7] == "<docno>":
            docnos.append(REPLACE_DOCNO.sub("",tokens[i][7:]))

        if fillbuf:
            # buf += REPLACE_TAG.sub("", tokens[i]) + " "
            buf += tokens[i]+" "


        if len(tokens[i]) > 1 and tokens[i][0] == "<" and tokens[i] == "</doc>":
            texts.append(buf)
            buf = ""
            fillbuf = False

    [print(t[:2000], "\n---") for t in texts]

    # return dict(zip(docnos, texts))

    # texts = []
    # buffer = ""
    # bufflag = False # track whether buffer is being filled or not
    # for line in doclist.splitlines():
    #     if line[:3] == "qid":
    #         texts.append(buffer)
    #         buffer = ""
    #         bufflag = False
    #         continue
    #
    #     if line[:5] == "<doc>":
    #         bufflag = True
    #
    #     if bufflag:
    #         buffer += "\n"+line
    #
    #     if line[:6] == "</doc>":
    #         bufflag = False
    #
    # print(texts[34])
    # dict = xmltodict.parse(texts[34])
    # print(dict)
    # # for i, s in enumerate(texts[1:]):
    # #     dict = xmltodict.parse(s)
    # #     print(i)
    #
    #


    return

"""
    parseTopDocs(data) returns a dictionary of questions numbers as keys and the topdoc id as the value
    Input : data -> the string that maps question number to topdoc id
    Output : xml_dict -> dictionary of key TopDoc integer id number and value string xml doc
"""
def parseTopDocs(data):

    # xml_dict = xmltodict.parse(data[0])
    # print(xml_dict)
    return

################################################################################

data = getData("training/qadata/", 0)
question_dict = parseQuestions(data[1])
# print(data[0])
# print(data[1])
# print(data[2])
id_dict = parseRelevantDocs(data[2])

topdoc_data = getData("training/topdocs/", 1)

# print(topdoc_data[4])

a = genXMLListFromTopDocs(topdoc_data[4])

# for k in a:
#     print(len(a))
#     print(k, "::", a[k][:1000])
