import os.path
import string
import sys
import math

class PositionalIndex:

    def __init__(self,doc_index,position):
        self.doc_index = doc_index
        self.termfrequency = 1
        self.lstposition = position


    def incrementDocFreqCount(self, position):
        self.termfrequency += 1
        self.lstposition.append(position)
        return True

    def appendEndOfFile(self):
        self.lstposition.append(float("inf"))

    def printIndex(self):
        print("document index",self.doc_index)
        print("term frequency", self.termfrequency)
        print("lstpositions",self.lstposition)


def getFileContents(filename):
    """
    Read the file and return it's contents
    :param filename:
    :return: contents of text file
    """

    if not os.path.exists(filename):
        print("File doesn't exisr")
        return

    fileText = open(filename)

    text = fileText.read()

    return text

def createPositionalIndex(corpus):
    docs = corpus.split('\n\n')
    dict_docs = {}
    num_of_docs = len(docs)
    term_dict = {}
    translator = str.maketrans(dict.fromkeys(string.punctuation))

    for doc_idx, doc in enumerate(docs):

        dict_docs[doc_idx] = doc
        words = doc.split()
        i = 0

        for word in words:
            word_without_punct = word.translate(translator)

            if word_without_punct not in term_dict:
                term_dict[word_without_punct] = [PositionalIndex(doc_idx, [float("-inf"),words.index(word,i)])]
            else:
                for pos_obj in term_dict[word_without_punct]:
                    if pos_obj.doc_index == doc_idx:
                        updated = pos_obj.incrementDocFreqCount(words.index(word, i))
                if not updated:
                    term_dict[word_without_punct].append(PositionalIndex(doc_idx, [float("inf"),words.index(word,i)]))
                updated = False

            i += 1

        for term in term_dict:
            for pos_obj in term_dict[term]:
                if pos_obj.doc_index == doc_idx :
                    pos_obj.appendEndOfFile()


    return num_of_docs,term_dict


def calcTFIDFDoc(term_dict,num_of_docs):
    doc_tfidf = {}
    
    for term in term_dict:
        for pos_obj in term_dict[term]:
            if pos_obj.doc_index not in doc_tfidf:
                doc_tfidf[pos_obj.doc_index] = [(term,calcTF(pos_obj.termfrequency)*calcIDF(num_of_docs,len(term_dict[term])))]
            else:
                doc_tfidf[pos_obj.doc_index].append((term,calcTF(pos_obj.termfrequency)*calcIDF(num_of_docs,len(term_dict[term]))))

    return doc_tfidf

def calcTF(term_frequency):
    return math.log(term_frequency,2)+1

def calcIDF(num_docs,term_doc_count):
    return math.log((num_docs/term_doc_count),2)

'''
def calcTFIDFQuery():
    query_terms = query.split()
    query_tfidf = {}
    translator = str.maketrans(dict.fromkeys(string.punctuation))

    for query_term in query_terms:
        
'''


def cosineRanking(corpus):

    (num_of_docs,term_dict) = createPositionalIndex(corpus)
    doc_tfidf = calcTFIDFDoc(term_dict,num_of_docs)

    #preethi's code here to determine which doc has all the query terms
    '''
    for term in term_dict:
        print(term)
        for pos_obj in term_dict[term]:
            pos_obj.printIndex()
    '''


filename = "/Users/raghavs/Desktop/SimpleText.txt"
#filename = sys.argv[1]
filecontents = getFileContents(filename)
query = "Information retrieval"
cosineRanking(filecontents)