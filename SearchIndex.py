import os.path
import string
import sys
import math

"""
To Run:
Cosine Ranking
Python3.6 SearchIndex.py path_to_corpus_file cos 5 “query” path_to_query_file

Proximity Ranking
Python3.6 SearchIndex.py path_to_corpus_file proximity 5 “query” 
"""

class InvertedIndex:
    def __init__(self, term_dict, num_docs, query_dict=None):
        self.term_dict = term_dict
        self.num_docs = num_docs
        self.query_dict = query_dict

    def first(self, t):
        if self.term_dict[t] is not None:
            return Current(self.term_dict[t][0].doc_index, self.term_dict[t][0].lstposition[0])
        return Current("inf", "inf")

    def last(self, t):
        if self.term_dict[t] is not None:
            return Current(self.term_dict[t][-1].doc_index, self.term_dict[t][-1].lstposition[-1])
        return Current("inf", "inf")

    def prev(self, t, current):
        """
            find prev occurence of the term t from given current position
            We use two iterations of gallop search to prev position in term posting list
            First gallop - finds doc_id_idx
            Second gallop - finds pos from doc searched by 1st iteration
            :param t - term
            :param current - current position
            :return previous position of t from current as (doc, pos) pair
            """
        if t not in self.term_dict:
            return Current("inf", "inf")
        else:
            # end of corpus -> returns last position in last doc
            if current.doc_id == "inf" and current.pos == "inf":
                return Current(self.term_dict[t][-1].doc_index, self.term_dict[t][-1].lstposition[-1])
            # finds doc_id
            doc_id_idx = galloping_search(self.term_dict[t], current.doc_id, True)

            if doc_id_idx == 'inf':
                return Current("inf", "inf")

            # current is not in searched term list, need to return last position from valid doc; need not find pos
            if self.term_dict[t][doc_id_idx].doc_index < current.doc_id:
                return Current(self.term_dict[t][doc_id_idx].doc_index, self.term_dict[t][doc_id_idx].lstposition[-1])

            # finds pos id
            pos_idx = galloping_search(self.term_dict[t][doc_id_idx].lstposition, current.pos)
            # term not found - so return current
            if self.term_dict[t][doc_id_idx].lstposition[pos_idx] < current.pos:
                return Current(self.term_dict[t][doc_id_idx].doc_index,
                               self.term_dict[t][doc_id_idx].lstposition[pos_idx])
            if pos_idx == 'inf':
                return Current("inf", "inf")

            # end of position list
            if pos_idx - 1 < 0:
                # end of doc list as well
                if doc_id_idx - 1 < 0:
                    return Current("inf", "inf")
                else:
                    # if not end of doclist but end of position list, return last element of prev doc
                    return Current(self.term_dict[t][doc_id_idx - 1].doc_index,
                                   self.term_dict[t][doc_id_idx - 1].lstposition[-1])
            else:
                # not end of doc list or pos list within it. normal case
                return Current(current.doc_id, self.term_dict[t][doc_id_idx].lstposition[pos_idx - 1])

    def next(self, t, current):
        """
        Find next instance of term from given 'current' position.
        We use two iterations of gallop search to next position in term posting list
        First gallop - finds doc_id_idx
        Second gallop - finds pos from doc searched by 1st iteration
        If current is present, galloping search returns it, and we find the next occurance
        If current is not present in positing list, search returns next nearest value and return that directly
        :param t - term
        :param current - current position
        :return next position of t from current as (doc, pos) pair
        """
        if t not in self.term_dict:
            return Current("inf", "inf")
        else:
            # find doc in which current is present
            doc_id_idx = galloping_search(self.term_dict[t], current.doc_id, True)
            if doc_id_idx == 'inf':
                return Current("inf", "inf")

            # current not found. next nearest is in next doc, so return first pos
            if self.term_dict[t][doc_id_idx].doc_index > current.doc_id:
                return Current(self.term_dict[t][doc_id_idx].doc_index, self.term_dict[t][doc_id_idx].lstposition[0])

            # find pos in given doc from prev gallop
            pos_idx = galloping_search(self.term_dict[t][doc_id_idx].lstposition, current.pos)

            # search finds a position before current
            if self.term_dict[t][doc_id_idx].lstposition[pos_idx] <= current.pos \
                    and self.term_dict[t][doc_id_idx].doc_index == current.doc_id:
                if pos_idx == 'inf':
                    return Current("inf", "inf")
                # position found by search is at the end of posting list
                if pos_idx + 1 == self.term_dict[t][doc_id_idx].termfrequency:
                    # end of corpus
                    if len(self.term_dict[t]) - 1 == doc_id_idx:
                        return Current("inf", "inf")
                    else:
                        # end of position list but not last doc
                        return Current(self.term_dict[t][doc_id_idx + 1].doc_index,
                                       self.term_dict[t][doc_id_idx + 1].lstposition[0])
                else:
                    # same doc and has more items in same doc list - normal case
                    return Current(current.doc_id, self.term_dict[t][doc_id_idx].lstposition[pos_idx + 1])
            elif current.pos < self.term_dict[t][doc_id_idx].lstposition[pos_idx]:
                # not found so default next nearest returned by galloping search is used as next
                return Current(self.term_dict[t][doc_id_idx].doc_index,
                               self.term_dict[t][doc_id_idx].lstposition[pos_idx])


class Current:
    """
    position of term represented as doc_id and position within that doc
    """

    def __init__(self, doc_id, pos):
        self.doc_id = doc_id
        self.pos = pos

    def print(self):
        print("doc_id = {}, pos= {} ", self.doc_id, self.pos)


class Cover:
    """
    Cover represented as pair of (doc, pos) each corresponding to leftmost and rightmost endpoints covering the given terms
    """

    def __init__(self, u_docid="inf", upos="inf", vpos=None):
        if vpos is not None:
            self.u.doc_id = u_docid
            self.v.doc_id = "inf"
            self.u.pos = upos
            self.v.pos = "inf"
        else:
            self.u = u_docid
            self.v = upos

    def printCover(self):
        print("cover :: ")
        print(self.u.doc_id, self.u.pos)
        print(self.v.doc_id, self.v.pos)

    # calculate (v - u + 1)
    def dist(self, posadd):
        return self.v.pos - self.u.pos + posadd


class PositionalIndex:
    """
    Every Position Index object has three members - doc_index, termfrequency, position in the document
    """

    def __init__(self, doc_index, position):
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
        print("document index", self.doc_index)
        print("term frequency", self.termfrequency)
        print("lstpositions", self.lstposition)


class Ranking:
    """
    This class is used to store results of ranking mathods (cosine, proximity) as (doc_id, score)
    """

    def __init__(self, doc_id, score):
        self.doc_id = doc_id
        self.score = score


"""
Helper functions
"""


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


def writeFileContents(filename, query, isfirstentry=False):
    """
    Write into the file specified by filename
    :param filename:
    :return:
    """
    if not os.path.exists(filename):
        print("File doesn't exist")
        return

    fileText = open(filename, "a")
    if not isfirstentry:
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
        corpus_doc_freq[doc_idx] = len(words)

        # i is used to keep track of position within the document
        i = 0
        updated = False

        # iterate over each word
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
                internal_term_dict[word_without_punct] = [PositionalIndex(doc_idx, [words.index(word, i)])]
            else:
                for pos_obj in internal_term_dict[word_without_punct]:
                    if pos_obj.doc_index == doc_idx:
                        updated = pos_obj.incrementDocFreqCount(words.index(word, i))
                if not updated:
                    internal_term_dict[word_without_punct].append(PositionalIndex(doc_idx, [words.index(word, i)]))
                updated = False

            i += 1

        '''
        After all the terms of the document is scanned, append the end-of-file indicator -inf into the posting list of 
        every term
        '''
        '''
        for term in internal_term_dict:
            for pos_obj in internal_term_dict[term]:
                if pos_obj.doc_index == doc_idx:
                    pos_obj.appendEndOfFile()
        '''

    return num_of_docs, internal_term_dict


def binary_search(lstposition, low, high, current_value, check_doc_id=False):
    """
    	:param lstposition: List of objects or values over which the galloping search has to be performed.
    	:param low: Lower limit for the binary search.
    	:param high: Higher limit for the binary search.
    	:param current_value: Value which will be searched in the lstposition list.
    	:param check_doc_id: Boolean for enabling flows within the galloping search.
    	:return: Position or index of the current_value within the lstposition.
    	    If the exact current_value is not found, next
    	nearest value's index is returned
    	"""
    # Checking the boolean to determine the flow - Flow for searching on a list of positions within a document index
    if not check_doc_id:
        # calculating the mid using low and high
        mid = int((low + high) / 2)
        if low < high:
            # if the position value at mid index is greater than the current_value, call the binary search with low, mid as high
            if lstposition[int(mid)] > current_value:
                return binary_search(lstposition, low, mid, current_value, check_doc_id)
            # if the position value at mid index is less than the current_value, call the binary search with mid as low, high
            elif lstposition[int(mid)] < current_value:
                return binary_search(lstposition, mid + 1, high, current_value, check_doc_id)
        # return mid if the value at mid position equals the current_value
        return int(mid)
    # Flow for searching the document id on a list of objects containing document id and positions
    else:
        # calculating the mid using low and high
        mid = int((low + high) / 2)
        if low < high:
            # if the document id at mid index is greater than the current_value, call the binary search with low, mid as high
            if lstposition[mid].doc_index > current_value:
                return binary_search(lstposition, low, mid, current_value, check_doc_id)
            # if the position value at mid index is less than the current_value, call the binary search with mid as low, high
            elif lstposition[mid].doc_index < current_value:
                return binary_search(lstposition, mid + 1, high, current_value, check_doc_id)
        return mid
    return mid


def galloping_search(lstposition, current_value, check_doc_id=False):
    """
    :param lstposition: List of objects or values over which the galloping search has to be performed.
    :param current_value: Value which will be searched in the lstposition list.
    :param check_doc_id: Boolean for enabling flows within the galloping search.
    :return: The value that is returned by the binary search.
    Galloping search method performs a general galloping (exponential) search over the given parameters. The search
    works by exponentially increasing the high value. When the value present at the high index position is less than
    the current value, the high value is multiplied by 2. In this case, the high value is set to low before high value
    increases exponentially. When the value present at the high index position is greater than the current_value,
    we fix the high value. After this, the binary search method is called with the high and low values.
    """
    # initialization
    low = 0
    jump = 1
    high = low + jump
    # Flag to determine the flow - for document id search
    if check_doc_id:
        # Iterate until the high value is less than the length of the document id list and
        # value at high is less than the current_value
        while (high < (len(lstposition) - 1) and lstposition[high].doc_index < current_value):
            low = high
            jump = 2 * jump
            high = low + jump
        # If high values is greater than the length of the object list, set high to the length of the object list
        if high > (len(lstposition) - 1):
            high = len(lstposition) - 1

    # Flag to determine the flow - for position index search
    else:
        # Iterate until the high value is less than the length of the position list and value at high is less than the current_value
        while (high < (len(lstposition) - 1) and lstposition[high] < current_value):
            low = high
            jump = 2 * jump
            high = low + jump
        # If high values is greater than the length of the position list, set high to the length of the position list
        if high > (len(lstposition) - 1):
            high = len(lstposition) - 1

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


def calcTFIDF(num_of_docs, doc_id, dict_docs, isdoccorpus):
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
                                                         calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(
                                                             internal_dict[term])))]
                        break
                    else:
                        doc_tfidf[pos_obj.doc_index].append((term,
                                                             calcTF(pos_obj.termfrequency) * calcIDF(num_of_docs, len(
                                                                 internal_dict[term]))))
                        break

    # for every TFIDF score of a term in the vector, normalize the vector
    for doc in doc_tfidf:
        magnitude = 0
        for (term, tfidf_val) in doc_tfidf[doc]:
            magnitude += (tfidf_val ** 2)
        magnitude = pow(magnitude, 0.5)
        if magnitude > 0:
            doc_tfidf[doc] = [(term, tfidf_val / magnitude) for (term, tfidf_val) in doc_tfidf[doc]]

    return doc_tfidf


def calcTF(term_frequency):
    """
    Calculate the TF score of the term
    :param term_frequency:
    :return:
    """
    return math.log(term_frequency, 2) + 1


def calcIDF(num_docs, term_doc_count):
    """
    Calculate the IDF score of the term
    :param num_docs:
    :param term_doc_count:
    :return:
    """
    return math.log((num_docs / term_doc_count), 2)


def nextDoc(term, doc_id, isdoccorpus):
    """
    :param term: the term whose occurence in the next document needs to be found
    :param doc_id: the current document id in which the term is present
    :param isdoccorpus: This boolean variable decides if we need to use positional index of the document corpus or
                        the positional index of the query corpus
    :return: the document_id of the document greater than doc_id in which the term appears
    This function uses the galloping method to set the higher and lower bounds for the nextDoc search
    """
    global term_dict
    global query_dict

    low = 0
    jump = 1
    high = low + jump

    if (isdoccorpus):
        local_dict = term_dict
    else:
        local_dict = query_dict

    if doc_id == float("-inf"):
        return local_dict[term][0]

    if local_dict[term][len(local_dict[term]) - 1].doc_index <= doc_id:
        return float("inf")

    else:
        while (high < len(local_dict[term]) - 1 and local_dict[term][high].doc_index <= doc_id):
            low = high
            jump = 2 * jump
            high = low + jump

        if high > len(local_dict[term]) - 1:
            high = len(local_dict[term]) - 1

        return binary_search_doc(local_dict[term], doc_id, low, high)


def binary_search_doc(lst_pos_obj, doc_id, low, high):
    """
    :param lst_pos_obj: The lst_pos_obj will contain the list of positional index objects
    :param doc_id: The current doc_id in which the term is present
    :param low: lower bound for the binary search
    :param high: higher bound for the binary search
    :return: Return the position of the positional object that has doc_index > doc_id
    """
    if low == high:
        return lst_pos_obj[low].doc_index
    if (high - low) == 1:
        if lst_pos_obj[low].doc_index > doc_id:
            return lst_pos_obj[low].doc_index
        elif lst_pos_obj[high].doc_index > doc_id:
            return lst_pos_obj[high].doc_index

    while (low < high):
        mid = (low + high) / 2
        mid = int(mid)

        if lst_pos_obj[mid].doc_index < doc_id:
            return binary_search_doc(lst_pos_obj, doc_id, int(mid), high)
        elif lst_pos_obj[mid].doc_index > doc_id:
            return binary_search_doc(lst_pos_obj, doc_id, low, mid)
        elif lst_pos_obj[mid].doc_index == doc_id:
            return lst_pos_obj[mid + 1].doc_index


def allTermsInDoc(query_terms, doc_id, isdoccorpus):
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
                next_doc_id = nextDoc(query_term, doc_id, isdoccorpus)
                if next_doc_id == float("inf"):
                    return float("inf")
                doc_allterms.append(next_doc_id)
                '''
                for pos_obj in pos_obj_list:
                    if pos_obj.doc_index > doc_id:
                        doc_allterms.append(pos_obj.doc_index)
                        break
                '''
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
        return allTermsInDoc(query_terms, min(doc_allterms), isdoccorpus)


def calcScore(doc_vector, query_vector, query_terms):
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
        for (term, tfidfscore) in doc_vector:
            if term == query_term:
                doc_scores.append(tfidfscore)
                break
        for (term, tfidfscore) in query_vector:
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

    for (docid, doc) in enumerate(corpus):
        dict_corpus[docid] = doc

    return dict_corpus


def isQueryinCorpus(dict_query, query):
    """
    Determine if the query given by the user is present in the query corpus or not
    :param dict_query:
    :param query:
    :return:
    """
    for query_id in dict_query:
        if dict_query[query_id].lower() == query.lower():
            return True

    return False


def cosineRanking(corpus, query_corpus, query, num_of_results):
    """
    Calculates the cosine ranking of a query against the document corpus,
    :param corpus:
    :param query_corpus:
    :param query:
    :return:
    """
    if not query_corpus:
        writeFileContents(queryfilename, query, True)
        query_corpus = query
    # for both document and query corpus , create the dictionaries with document index as id and document as the value.
    dict_docs = createCorpusDict(corpus.split('\n\n'))
    dict_query = createCorpusDict(query_corpus.split('\n\n'))

    # if the query is not present in the corpus, add it to the corpus, the dict_query and the query corpus file
    if not isQueryinCorpus(dict_query, query):
        dict_query[len(dict_query)] = query
        writeFileContents(queryfilename, query)
        query_corpus = getFileContents(queryfilename)

    # create positional index of both the document and query corpus
    global term_dict
    global query_dict
    (num_of_docs, term_dict) = createPositionalIndex(corpus)
    (num_of_queries, query_dict) = createPositionalIndex(query_corpus)

    cosine_results = []
    # Split the query terms and remove punctuations from them and convert them to lower case
    query_terms = query.split()
    query_terms = [Removepunctuation(query_term).lower() for query_term in query_terms]

    # find the document which has all the query terms
    next_doc_id = allTermsInDoc(query_terms, float("-inf"), True)
    # find the query within the query corpus. The next_query_id is the index of the query in the query corpus.
    next_query_id = allTermsInDoc(query_terms, float("-inf"), False)

    if next_doc_id != float("-inf"):
        # query_tfidf = calcTFIDFQuery(query_dict,num_of_queries, query_terms)
        # Calculate the query TFIDF values. This is the query vector
        query_tfidf = calcTFIDF(num_of_queries, next_query_id, dict_query, False)

    while (next_doc_id < float("inf")):
        # Calculate the TFIDF values for all the terms in the document that has all the query terms.
        # This is the document vector
        doc_tfidf = calcTFIDF(num_of_docs, next_doc_id, dict_docs, True)

        # Calculate the cosine score using the document vector and the query vector
        score = calcScore(doc_tfidf[next_doc_id], query_tfidf[next_query_id], query_terms)
        cosine_results.append(Ranking(next_doc_id, score))

        # calculate the TFIDF values for the terms in the next document that contains all the query terms.
        next_doc_id = allTermsInDoc(query_terms, next_doc_id, True)

    if cosine_results:
        cosine_results.sort(key=lambda x: x.score, reverse=True)
        results = cosine_results[:int(num_of_results)]
        print("DocId\tScore")
        for result in results:
            print(result.doc_id,'\t',result.score)

    if not cosine_results:
        print("No document has all the query terms")


"""
Finds minimum of given list of (doc, positional lists) pairs.
if end of corpus (inf, inf)
"""


def min_current(currents):
    min_doc = Current("inf", "inf")
    for cur in currents:
        if min_doc.doc_id == "inf":
            min_doc = cur
        if min_doc.doc_id > cur.doc_id:
            min_doc = cur
        elif (min_doc.doc_id == cur.doc_id) and (min_doc.pos > cur.pos):
            min_doc = cur
    return min_doc


"""
Finds maximum of given list of (doc, positional lists) pairs.
if end of corpus (inf, inf)
"""


def max_current(currents):
    max_doc = Current(-1, -1)
    for cur in currents:
        if cur.doc_id == "inf":
            return Current("inf", "inf")
        if max_doc.doc_id < cur.doc_id:
            max_doc = cur
        elif (max_doc.doc_id == cur.doc_id):
            if (max_doc.pos <= cur.pos):
                max_doc = cur
    return max_doc


"""
Finds next cover of query terms given corpus and position.
:param terms: query terms
:param position current position to start searching for cover
:param adt - has the inverted index adt 
"""


def nextCover(terms, position, adt):
    nextv = []
    nextu = []
    for term in terms:
        nextv.append(adt.next(term, position))
    v = max_current(nextv)
    if v.doc_id == "inf" or v.pos == "inf":
        return Cover("inf", "inf")
    for term in terms:
        # next word in the corpus after v; finds v+1
        v_1 = nextCorpusWord(v)
        nextu.append(adt.prev(term, v_1))
    u = min_current(nextu)

    if u.doc_id == v.doc_id:
        myc = Cover(u, v)
        return myc
    else:
        return nextCover(terms, u, adt)


"""
Finds the ranking for each cover
"""


def rankProximity(terms, adt):
    result = []
    uv = nextCover(terms, Current(0, 0), adt)
    if uv.u == "inf" and uv.v == "inf":
        print("No cover found")
        return result
    d = uv.u.doc_id
    score = 0
    while uv.u != "inf":
        if d < uv.u.doc_id:
            result.append(Ranking(d, score))
            d = uv.u.doc_id
            score = 0
        score = score + 1 / uv.dist(1)
        uv = nextCover(terms, uv.u, adt)
    if d != "inf":
        result.append(Ranking(d, score))
    # sort results
    result.sort(key=lambda x: x.score, reverse=True)
    return result


def nextCorpusWord(v):
    v_len = corpus_doc_freq[v.doc_id]
    if (v.pos + 1 < v_len):
        # one more term present in same doc
        return Current(v.doc_id, v.pos + 1)
    elif (v.pos + 1 == v_len):
        # not end of corpus but end of current document, return first pos in next doc
        if (len(corpus_doc_freq) - 2) > v.doc_id:
            return Current(v.doc_id + 1, 0)
        else:
            # end of corpus
            return Current("inf", "inf")
    else:
        # invalid position
        return Current("inf", "inf")


def proximityRanking(corpus, query, num_of_results):
    (num_of_docs, term_dict) = createPositionalIndex(corpus)
    adt = InvertedIndex(term_dict, num_of_docs)
    rankings = rankProximity(query.lower().split(), adt)
    truncated = rankings[:int(num_of_results)]
    if len(truncated) > 0:
        print("DocId\tScore")
    for r in truncated:
        print(r.doc_id, "\t", r.score)


term_dict = {}
query_dict = {}
filename = sys.argv[1]
filecontents = getFileContents(filename)
ranking = sys.argv[2]
num_of_results = sys.argv[3]
query = sys.argv[4]

# To store the corpus length for each document
corpus_doc_freq = {-1: -1}

if ranking == 'proximity':
    proximityRanking(filecontents, query, num_of_results)

if ranking == 'cos':
    queryfilename = sys.argv[5]
    queryfilecontents = getFileContents(queryfilename)
    cosineRanking(filecontents, queryfilecontents, query, num_of_results)

