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

class Ranking:
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
        print("File doesn't exist")
        return

    fileText = open(filename)

    text = fileText.read()

    return text

def writeFileContents(filename,query):
    """
    Write into the file specified by filename
    :param filename:
    :return:
    """
    if not os.path.exists(filename):
        print("Could not add query")
        return

    fileText = open(filename,"a")
    fileText.write('\n\n')
    fileText.write(query)

    return True

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
    internal_term_dict = {}

    # iterate over each document
    for doc_idx, doc in enumerate(docs):

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
            if word_without_punct not in internal_term_dict:
                internal_term_dict[word_without_punct] = [PositionalIndex(doc_idx, [float("-inf"),words.index(word,i)])]
            else:
                for pos_obj in internal_term_dict[word_without_punct]:
                    if pos_obj.doc_index == doc_idx:
                        updated = pos_obj.incrementDocFreqCount(words.index(word, i))
                if not updated:
                    internal_term_dict[word_without_punct].append(PositionalIndex(doc_idx, [float("inf"),words.index(word,i)]))
                updated = False

            i += 1

        '''
        After all the terms of the document is scanned, append the end-of-file indicator -inf into the posting list of 
        every term
        '''
        for term in internal_term_dict:
            for pos_obj in internal_term_dict[term]:
                if pos_obj.doc_index == doc_idx:
                    pos_obj.appendEndOfFile()


    return num_of_docs,internal_term_dict

def binary_search(lstposition, low, high, current_value, check_doc_id = False):
        if(high-low) == 1:
            return low
        mid = low + (high - low)/2
        if not check_doc_id:
            if lstposition[int(mid)] > current_value:
                if mid == low:
                    return float('inf')
                return binary_search(lstposition, low, mid-1, current_value, check_doc_id)
            elif lstposition[int(mid)] < current_value:
                if mid == high:
                    return float('inf')
                return binary_search(lstposition, mid+1, high, current_value, check_doc_id)
            else:
                return int(mid)
        else:
            if lstposition[int(mid)].doc_index > current_value:
                if mid == low:
                    return float('inf')
                return binary_search(lstposition, low, mid-1, current_value, check_doc_id)
            elif lstposition[int(mid)].doc_index < current_value:
                if mid == high:
                    return float('inf')
                return binary_search(lstposition, mid+1, high, current_value, check_doc_id)
            else:
                return int(mid)


def galloping_search(lstposition, current_value, check_doc_id = False):
        low = 0
        jump = 1
        while(True):
            high = low + jump
            if high > len(lstposition)-1:
                high = len(lstposition)-1
                break
            if not check_doc_id and lstposition[high] > current_value:
                break
            if check_doc_id and lstposition[high].doc_index > current_value:
                break
            else:
                jump = jump * 2
        return binary_search(lstposition, low, high, current_value, check_doc_id)


def Removepunctuation(term):
    """
        Removes the punctuations from a term using the translator
    :param term:
    :return:
    """

    translator = str.maketrans(dict.fromkeys(string.punctuation))
    term_no_punct = term.translate(translator)
    return term_no_punct
'''
def calcTFIDFQuery(term_dict, num_of_docs, queryterms):

    query_tfidf = {}

    for queryterm in queryterms:
        if queryterm in term_dict:
            term_frequency = queryterms.count(queryterm)
            query_tfidf[queryterm] = calcTF(term_frequency) * calcIDF(num_of_docs, len(term_dict[queryterm]))
        else:
            query_tfidf[queryterm] = 0

    magnitude = 0
    for queryterm in query_tfidf:
        magnitude += (query_tfidf[queryterm] ** 2)

    for queryterm in query_tfidf:
        query_tfidf[queryterm] /= magnitude

    return query_tfidf
'''

def calcTFIDF(num_of_docs,doc_id,dict_docs,isdoccorpus):
    """
    For every term in the document where all the query terms are found, calculate the tfidf score. Use dict_docs to
    access every term in the document. Use the term_dict to access the positional index information about the term.
    Using the term_dict we can find the frequency of the term in the given document.
    return a dictionary with key as document id. Value is a tuple-(term,TFIDF_score)
    :param term_dict: Positional index of the corpus
    :param num_of_docs: Number of documents in the corpus
    :param doc_id: document id where all the query terms are found
    :param dict_docs: dictionary where doc_id is the key and the content of the document is the value.
    :return: doc_tfidf: return a dictionary with key as document id. Value is a tuple-(term,TFIDF_score)
    """
    doc_tfidf = {}
    global term_dict
    global query_dict
    if isdoccorpus:
        internal_dict = term_dict
    else:
        internal_dict = query_dict

    terms_in_doc = dict_docs[doc_id].split()

    for term in terms_in_doc:
        term = Removepunctuation(term).lower()
        if term in internal_dict:

            for pos_obj in internal_dict[term]:
                if pos_obj.doc_index == doc_id:
                    if pos_obj.doc_index not in doc_tfidf:
                        doc_tfidf[pos_obj.doc_index] = [(term,
                                                         calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(internal_dict[term])))]
                        break
                    else:
                        doc_tfidf[pos_obj.doc_index].append((term,
                                                             calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(internal_dict[term]))))
                        break


    # for every TFIDF score of a term in the vector, normalize the vector
    for doc in doc_tfidf:
        magnitude = 0
        for (term, tfidf_val) in doc_tfidf[doc]:
            magnitude += (tfidf_val ** 2)
        magnitude = pow(magnitude, 0.5)

        doc_tfidf[doc] = [(term,tfidf_val/magnitude) for (term,tfidf_val) in doc_tfidf[doc]]

    return doc_tfidf

def calcTF(term_frequency):
    """
    Calculate the TF score of the term
    :param term_frequency:
    :return:
    """
    return math.log(term_frequency,2)+1

def calcIDF(num_docs,term_doc_count):
    """
    Calculate the IDF score of the term
    :param num_docs:
    :param term_doc_count:
    :return:
    """
    return math.log((num_docs/term_doc_count),2)


def allTermsInDoc(query_terms,doc_id,isdoccorpus):
    """
    This function returns the document id of the document that has all the query terms. For every query_term, based on
    the doc_id, we find the positional index of the term in a document > doc_id. We use doc_terms to store the
    document id where the next instance of each query term is found(after the given doc_id). If at the end, the
    doc_terms has the same document ids, then that document has all the query terms and hence that document id is
    returned. If only partial query terms are found in a document, we call the same function with least value of
    document id found in doc_terms. If all the query terms are not found in the entire corpus, we return inf.
    :param query_terms:
    :param term_dict:
    :param doc_id:
    :return:
    """

    doc_allterms = []
    global term_dict
    global query_dict
    if isdoccorpus:
        internal_dictionary = term_dict
    else:
        internal_dictionary = query_dict
    
    for query_term in query_terms:
        if query_term in internal_dictionary:
            if doc_id == float("-inf"):
                pos_obj = internal_dictionary[query_term][0]
                doc_allterms.append(pos_obj.doc_index)
            else:
                pos_obj_list = internal_dictionary[query_term]
                for pos_obj in pos_obj_list:
                    if pos_obj.doc_index > doc_id:
                        doc_allterms.append(pos_obj.doc_index)
                        break
                '''
                if len(internal_dictionary[query_term]) == 1:
                    pos_obj = internal_dictionary[query_term][0]
                    doc_allterms.append(pos_obj.doc_index)
                else:
                    doc_index_pos = galloping_search(internal_dictionary[query_term],doc_id,True)
                    if doc_index_pos < len(internal_dictionary[query_term])-1:
                        pos_obj = internal_dictionary[query_term][doc_index_pos+1]
                        doc_allterms.append(pos_obj.doc_index)
                    elif doc_index_pos == len(internal_dictionary[query_term])-1:
                        return float("inf")
                '''
        else:
            return float("inf")

    if not doc_allterms:
        return float("inf")
    if all(len(doc_allterms) == len(query_terms) and (doc_id == doc_allterms[0]) for doc_id in doc_allterms):
        return doc_allterms[0]
    else:
        return allTermsInDoc(query_terms,min(doc_allterms),isdoccorpus)

def calcScore(doc_vector,query_vector,query_terms):
    """
    Calculate the cosine similarity score for the given query and the document that has all the query terms.
    It is a simple multiplication between the terms of the vectors that are normalized.
    :param doc_vector:
    :param query_vector:
    :param query_terms:
    :return:
    """

    doc_scores = []
    queryterm_score = []
    for query_term in query_terms:
        for (term,tfidfscore) in doc_vector:
            if term == query_term:
                doc_scores.append(tfidfscore)
                break
        for (term,tfidfscore) in query_vector:
            if term == query_term:
                queryterm_score.append(tfidfscore)
                break
    score = 0
    for i in range(len(doc_scores)):
        score += (doc_scores[i] * queryterm_score[i])

    return score

def createCorpusDict(corpus):
    """
    Create and return a dictionary with document id as key and entire document as the value. This is called for both
    the document corpus and query corpus.
    :param corpus:
    :return:
    """
    dict_corpus = {}

    for (docid,doc) in enumerate(corpus):
        dict_corpus[docid] = doc

    return dict_corpus

def isQueryinCorpus(dict_query,query):
    """
    Determine if the query given by the user is present in the query corpus or not
    :param dict_query:
    :param query:
    :return:
    """
    for query_id in dict_query:
        if dict_query[query_id] == query:
            return True

    return False

def cosineRanking(corpus,query_corpus,query,num_of_results):
    """
    Calculates the cosine ranking of a query against the document corpus,
    :param corpus:
    :param query_corpus:
    :param query:
    :return:
    """
    # for both document and query corpus , create the dictionaries with document index as id and document as the value.
    dict_docs = createCorpusDict(corpus.split('\n\n'))
    dict_query = createCorpusDict(query_corpus.split('\n\n'))

    # if the query is not present in the corpus, add it to the corpus, the dict_query and the query corpus file
    if not isQueryinCorpus(dict_query,query):
        dict_query[len(dict_query)] = query
        writeFileContents(queryfilename,query)
        query_corpus = getFileContents(queryfilename)

    # create positional index of both the document and query corpus
    global term_dict
    global query_dict
    (num_of_docs,term_dict) = createPositionalIndex(corpus)
    (num_of_queries,query_dict) = createPositionalIndex(query_corpus)

    cosine_results = []
    # Split the query terms and remove punctuations from them and convert them to lower case
    query_terms = query.split()
    query_terms = [Removepunctuation(query_term).lower() for query_term in query_terms]

    # find the document which has all the query terms
    next_doc_id = allTermsInDoc(query_terms,float("-inf"),True)
    # find the query within the query corpus. The next_query_id is the index of the query in the query corpus.
    next_query_id = allTermsInDoc(query_terms,float("-inf"),False)

    if next_doc_id != float("-inf"):
        #query_tfidf = calcTFIDFQuery(query_dict,num_of_queries, query_terms)
        # Calculate the query TFIDF values. This is the query vector
        query_tfidf = calcTFIDF(num_of_queries,next_query_id,dict_query,False)


    while(next_doc_id < float("inf")):
        # Calculate the TFIDF values for all the terms in the document that has all the query terms.
        # This is the document vector
        doc_tfidf = calcTFIDF(num_of_docs, next_doc_id,dict_docs,True)

        # Calculate the cosine score using the document vector and the query vector
        score = calcScore(doc_tfidf[next_doc_id],query_tfidf[next_query_id],query_terms)
        cosine_results.append(Ranking(next_doc_id,score))

        #calculate the TFIDF values for the terms in the next document that contains all the query terms.
        next_doc_id = allTermsInDoc(query_terms,next_doc_id,True)

    for i in range(0,int(num_of_results)):
        if i < len(cosine_results):
            print (cosine_results[i].doc_id,cosine_results[i].score)


#filename = "/Users/raghavs/Desktop/Corpus.txt"
term_dict = {}
query_dict = {}
filename = sys.argv[1]
filecontents = getFileContents(filename)
#queryfilename = "/Users/raghavs/Desktop/Query.txt"
ranking = sys.argv[2]
num_of_results = sys.argv[3]
query = sys.argv[4]
#ranking = 'cos'
#num_of_results = 5

#query = "San Clara County"
if ranking == 'cos':
    queryfilename = sys.argv[5]
    queryfilecontents = getFileContents(queryfilename)
    cosineRanking(filecontents,queryfilecontents,query,num_of_results)


