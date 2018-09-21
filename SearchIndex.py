import os.path
import string
import sys
import math

class PositionalIndex:
    """
    Every Position Index object has three members - doc_index, termfrequency, position in the document
    """

    def __init__(self,doc_index,position):
        self.doc_index = doc_index
        self.termfrequency = 1
        self.lstposition = position

    # if the positional index object for the given doc is already present, only the frequency of the term has to
    # be updated
    def incrementDocFreqCount(self, position):
        self.termfrequency += 1
        self.lstposition.append(position)
        return True

    # posting list of every term should end with infinity
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
    """
    For every term in the corpus, a list of positional index objects. Each positional index object contains the
    doc_id in which the term is present, frequency of the term in that document, position of the term in that document
    :param corpus:
    :return: positional index for the corpus
    """
    # entire corpus file is split into multiple documents
    docs = corpus.split('\n\n')
    dict_docs = {}
    num_of_docs = len(docs)
    # term_dict contains the positional index of the entire corpus. It is a dictionary with key as the term, the value
    # is the list of positional index objects
    term_dict = {}
    translator = str.maketrans(dict.fromkeys(string.punctuation))

    # iterate over each document
    for doc_idx, doc in enumerate(docs):

        dict_docs[doc_idx] = doc
        words = doc.split()
        # i is used to keep track of position within the document
        i = 0
        updated = False

        #iterate over each word
        for word in words:
            # remove the punctuations
            word_without_punct = word.translate(translator)
            # change the case of every term to lower case
            word_without_punct = word_without_punct.lower()
            '''
            if the term is not present in term_dict, create a new positional index object. Fill the position of
            the term beginning with -inf and then the position of first occurrence of the term.
             if the term is present in the dictionary, check if the entry for the current doc_id is already present.
            If present, then just increment the term frequency. Else, create a new positional index object and 
            append it to the list which is the value of the dictionary
            '''
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

        '''
        After all the terms of the document is scanned, append the end-of-file indicator -inf into the posting list of 
        every term
        '''
        for term in term_dict:
            for pos_obj in term_dict[term]:
                if pos_obj.doc_index == doc_idx:
                    pos_obj.appendEndOfFile()


    return num_of_docs,term_dict


def calcTFIDF(term_dict,num_of_docs):
    doc_tfidf = {}

    for term in term_dict:
        for pos_obj in term_dict[term]:
            if pos_obj.doc_index not in doc_tfidf:
                doc_tfidf[pos_obj.doc_index] = [(term,calcTF(pos_obj.termfrequency)*calcIDF(num_of_docs,len(term_dict[term])))]
            else:
                doc_tfidf[pos_obj.doc_index].append((term,calcTF(pos_obj.termfrequency)*calcIDF(num_of_docs,len(term_dict[term]))))

    for doc in doc_tfidf:
        magnitude = 0
        for (term, tfidf_val) in doc_tfidf[doc]:
            magnitude += (tfidf_val ** 2)
        magnitude = pow(magnitude, 0.5)

        doc_tfidf[doc] = [(term,tfidf_val/magnitude) for (term,tfidf_val) in doc_tfidf[doc]]

    return doc_tfidf

def calcTF(term_frequency):
    return math.log(term_frequency,2)+1

def calcIDF(num_docs,term_doc_count):
    return math.log((num_docs/term_doc_count),2)


def cosineRanking(corpus,query_corpus,query):

    (num_of_docs,term_dict) = createPositionalIndex(corpus)
    (num_of_queries,query_dict) = createPositionalIndex(query_corpus)
    doc_tfidf = calcTFIDF(term_dict,num_of_docs)
    query_tfidf = calcTFIDF(query_dict,num_of_queries)

    query_terms = query.split()

    #preethi's code here to determine which doc has all the query terms - Use the query_terms



filename = "/Users/raghavs/Desktop/Corpus.txt"
#filename = sys.argv[1]
filecontents = getFileContents(filename)
queryfilename = "/Users/raghavs/Desktop/Query.txt"
queryfilecontents = getFileContents(queryfilename)
query = "San Jose city"
cosineRanking(filecontents,queryfilecontents,query)