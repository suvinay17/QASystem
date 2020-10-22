import numpy as np
import glob
import re
import os


def getData(path):
    files = glob.glob(os.path.join(os.getcwd(), path, "*.txt"))
    data = []
    for file in files:
        with open(file) as text:
            data.append(text.read().lower()) # this converts data to lower case
    return data


def parseQuestions(data):
    lines = data.splitlines()
    question_dict = {}
    for i in range(0, len(lines), 3):
        temp = lines[i].split()
        question_dict[lines[i+1]] = temp[1]
    return question_dict

    
data = getData("training/qadata/")
parseQuestions(data[0])
