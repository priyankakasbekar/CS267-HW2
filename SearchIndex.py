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

class CosineRanking:
    """
    This class is used to store cosine results
    """
    def __init__(self,doc_id,score):
        self.doc_id = doc_id
        self.score = score


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

        #dict_docs[doc_idx] = doc
        words = doc.split()
        # i is used to keep track of position within the document
        i = 0
        updated = False

        #iterate over each word
        for word in words:
            # remove the punctuations
            word_without_punct = Removepunctuation(word)
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
'''

def calcTFIDF(term_dict,num_of_docs,doc_id):
    doc_tfidf = {}

    for term in term_dict:
        for pos_obj in term_dict[term]:
            if pos_obj.doc_index == doc_id:
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
'''

def Removepunctuation(term):

    translator = str.maketrans(dict.fromkeys(string.punctuation))
    term_no_punct = term.translate(translator)
    return term_no_punct


def calcTFIDF(term_dict,num_of_docs,doc_id,dict_docs):
    doc_tfidf = {}

    terms_in_doc = dict_docs[doc_id].split()

    for term in terms_in_doc:
        term = Removepunctuation(term).lower()
        if term in term_dict:
            for pos_obj in term_dict[term]:
                if pos_obj.doc_index == doc_id:
                    if pos_obj.doc_index not in doc_tfidf:
                        doc_tfidf[pos_obj.doc_index] = [(term,
                                                         calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(term_dict[term])))]
                        break
                    else:
                        doc_tfidf[pos_obj.doc_index].append((term,
                                                             calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(term_dict[term]))))
                        break


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

def allTermsInDoc(query_terms,term_dict,doc_id):

    doc_allterms = []
    
    for query_term in query_terms:
        if query_term in term_dict:
            if doc_id == float("-inf"):
                pos_obj = term_dict[query_term][0]
                doc_allterms.append(pos_obj.doc_index)
            else:
                pos_obj_list = term_dict[query_term]
                for pos_obj in pos_obj_list:
                    if pos_obj.doc_index > doc_id:
                        doc_allterms.append(pos_obj.doc_index)
                        break
        else:
            return float("inf")

    if not doc_allterms:
        return float("inf")
    if all(len(doc_allterms) == len(query_terms) and (doc_id == doc_allterms[0]) for doc_id in doc_allterms):
        return doc_allterms[0]
    else:
        return allTermsInDoc(query_terms,term_dict,doc_allterms[0])

def calcScore(doc_vector,query_vector,query_terms):

    score = 0

    for query_term in query_terms:
        doc_scores = [score for (term,score) in doc_vector if(term == query_term)]
        queryterm_score = [score for (term,score) in query_vector if (term == query_term)]
        score = sum(docscore * queryscore for docscore in doc_scores for queryscore in queryterm_score)

    return score

def createCorpusDict(corpus):
    dict_corpus = {}

    for (docid,doc) in enumerate(corpus):
        dict_corpus[docid] = doc

    return dict_corpus


def cosineRanking(corpus,query_corpus,query):

    dict_docs = createCorpusDict(corpus.split('\n\n'))
    dict_query = createCorpusDict(query_corpus.split('\n\n'))

    (num_of_docs,term_dict) = createPositionalIndex(corpus)
    (num_of_queries,query_dict) = createPositionalIndex(query_corpus)

    cosine_results = []
    query_terms = query.split()
    query_terms = [query_term.lower() for query_term in query_terms]

    next_doc_id = allTermsInDoc(query_terms,term_dict,float("-inf"))
    next_query_id = allTermsInDoc(query_terms,query_dict,float("-inf"))

    if next_doc_id != float("-inf"):
        query_tfidf = calcTFIDF(query_dict, num_of_queries,next_query_id,dict_query)

    while(next_doc_id < float("inf")):
        doc_tfidf = calcTFIDF(term_dict, num_of_docs, next_doc_id,dict_docs)

        score = calcScore(doc_tfidf[next_doc_id],query_tfidf[next_query_id],query_terms)
        cosine_results.append(CosineRanking(next_doc_id,score))

        next_doc_id = allTermsInDoc(query_terms,term_dict,next_doc_id)

    for result in cosine_results:
        print (result.doc_id,result.score)



filename = "/Users/raghavs/Desktop/Corpus.txt"
#filename = sys.argv[1]
filecontents = getFileContents(filename)
queryfilename = "/Users/raghavs/Desktop/Query.txt"
queryfilecontents = getFileContents(queryfilename)
query = "San Jose"
cosineRanking(filecontents,queryfilecontents,query)