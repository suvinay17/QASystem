import numpy as np
import glob
import re
import os


"""
    getData(path) returns a list of strings where each string is .txt files in a given path
    Input : path -> the directory from where the .txt files are read
    Outout : data -> list of strings
"""
def getData(path):
    files = glob.glob(os.path.join(os.getcwd(), path, "*.txt"))
    data = []
    for file in files:
        with open(file) as text:
            data.append(text.read().lower()) # this converts data to lower case
    return data


def getXmlData(path):
    data = []
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


"""
    parseTopDocs(data) returns a dictionary of topdoc ids as keys and the topdoc string as the value
    Input : data -> list of the topdoc strings for each .txt file in topdocs
    Outout : topdoc_dict -> dictionary of key integer topdoc id and value string topdoc
"""
def parseTopDocs(data):
    for datum in data:
        print(datum)
    return


data = getData("training/qadata/")
question_dict = parseQuestions(data[0])
id_dict = parseRelevantDocs(data[1])
topdoc_data = getXmlData("training/topdocs/")
