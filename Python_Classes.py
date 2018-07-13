
# coding: utf-8

# # Python Classes
#
# In this notebook we will create a simplified version of the CountVectorizer class from scikit-learn as a means to explain Python classes.
#
# CountVectorizer converts a collection of text documents to a matrix of token counts. In this, it is a utility to vectorize unstructured text by counting the number of times that each token shows up in a corpora of documents.

from sklearn.feature_extraction.text import CountVectorizer


# create a list of documents
text = [
      'This is the first document'
    , 'This is the second second document'
    , 'And the third one'
    , 'Is it the first document again'
]

# ### Observe scikit-learn CountVectorizer to determine what we need to recreate
# - CountVectorizer acts as a blueprint to create different versions of CountVectorizer
# - We start by creating one instance of CountVectorizer, which we have decided to call vect, using the default parameters
# - Let's observe some of the available attributes and methods (terminology we will discuss later) in CountVectorizer

# create an instance of countvectorizer
vect = CountVectorizer()

# when we print vect, we see its hyperparameters
print(vect)

# The vectorizer learns the vocabulary when we fit it with our documents.
# This means it learns the distinct tokens (terms) in the text of the documents.
# We can observe these with the method get_feature_names

vect.fit(text)
vect.get_feature_names()

# Transform creates a sparse matrix, identifying the indices where terms are stores in each document
# This sparse matrix has 4 rows and 9 columns

vect.transform(text)

# This is easier to understand when we covert the sparse matrix into a dense matrix or pandas DataFrame
vect.transform(text).toarray()

import pandas as pd
data = vect.transform(text).toarray()
columns = vect.get_feature_names()
pd.DataFrame(data, columns=columns)

# by instantiating CountVectorizer with differnt parameters, we can change the vocabulary
# lowercase determines if all words should be lowercase, setting it to False includes uppercase words

vect = CountVectorizer(lowercase=False)
vect.fit(text)
vect.get_feature_names()

# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents
vect = CountVectorizer(stop_words='english')
vect.fit(text)
vect.get_feature_names()

# max features determines the maximum number of features to display
vect = CountVectorizer(max_features=5)
vect.fit(text)
vect.get_feature_names()


# ### Create functions to replicate CountVectorizer
# #### The are many other methods and parameters, but lets begin with the following few:
# #### Methods
# - fit
# - get_feature_names
# - transform
#
# #### Attributes
# - lowercase
# - stop_words
# - max_features

# ### Create a fit function, to recreate the fit functionality from CountVectorizer

def fit(raw_documents):
    """
    :param raw_documents: iterable over raw text documents

    :return sorted_tokens: tokens sorted
    """

    # combine all of the raw_documents into a string, separated by spaces
    combined_sentences = ' '.join(raw_documents)

    # separate the string into individual tokens (terms), split the overall string by spaces
    all_tokens = combined_sentences.split(' ')

    # only keep the set of distinct tokens (i.e. do not keep multiple copies of a word)
    distinct_tokens = set(all_tokens)

    # sort the terms alphabetically
    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text)
print(sorted_tokens)


# ### Implement a lowercase parameter in fit
def fit(raw_documents, lowercase=False):
    """
    :param raw_documents: iterable over raw text documents
    :param lowercase: boolean, default=True
        Convert all characters to lowercase before tokenizing

    :return sorted_tokens: tokens sorted
    """

    # add a check for the lowercase parameter
    # convert all documents to lowercase
    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')
    distinct_tokens = set(all_tokens)
    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, lowercase=True)
print(sorted_tokens)

# observe the new list comprehension to lowercase the original documents
print('Original documents: {} \n'.format(text))

print('lowercase documents: {}'.format([doc.lower() for doc in text]))


# ### Implement a stop_words parameter in fit
def fit(raw_documents, stop_words):
    """
    :param raw_documents: iterable over raw text documents
    :param stop_words: string {'english'} or list
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    :return sorted_tokens: tokens sorted by frequency
    """

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')

    # add a check for the stop_words parameter
    # remove all words that are in the stops_words list
    # otherwise keep all distinct tokens
    if stop_words:
        distinct_tokens = [token for token in set(all_tokens)
                           if token not in stop_words]
    else:
        distinct_tokens = set(all_tokens)

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, stop_words=['a','of'])
print(sorted_tokens)

# stop_words often include many tokens. Typing them each time leaves room for error.
print(fit(text, stop_words=['A','Another']))

# It would get tedious to add in a list of stops words ourselves every time we use the fit function. Also we may forget to add the same words every time or miss a couple words in a long list.
#
# To avoid this we can add a static list of stop words inside of the function.

# add a default list of stop words

def fit(raw_documents, stop_words=None):
    """
    :param raw_documents: iterable over raw text documents
    :param stop_words: string {'english'} or list
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    :return sorted_tokens: tokens sorted
    """

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')

    # add a check if you add your own stop words
    if stop_words == 'english':
        # include a default list of stop words in the function
        stop_words = ['a','of']

    if stop_words:
        distinct_tokens = [token for token in set(all_tokens)
                           if token not in stop_words]
    else:
        distinct_tokens = set(all_tokens)

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, stop_words=None)
print(sorted_tokens)


# ### Implement max features paramater to fit
#
# - max features return the top most common terms; thus, we need to determine the word count for each token
# - We could accomplish this in the fit function, but it makes sense to split it into its own function as we may need to use the output in multiple places (e.g. use token_stats for fit and transform)
from collections import defaultdict

s = 'mississippi'

d = defaultdict(int)
for k in s:
    d[k] += 1

d.items()


# As we will be adding many unseen terms to a dict and counting their frequency across documents, defaultdict is useful.
# defaultdict adds to dict functionality, allowing us to simplify the code by avoiding a check if an instance has been added to a dict before incrementing its count.
# - defaultdict examples: https://www.accelebrate.com/blog/using-defaultdict-python/

def get_token_stats(raw_documents):
    """
    :param raw_documents: iterable over raw text documents

    :return frequent_tokens: list tokens sorted by frequency
    """

    # set an empty dict to store each token and its count e.g. {'token1': 1, 'token2': 2}
    token_stats = defaultdict(int)

    # iterate through all documents, then iterate through all terms
    # increase the count for a token by one each time it appears in any document
    for doc in raw_documents:
        for term in doc.split(' '):
            token_stats[term] += 1

    # create a list of the tokens sorted by occurance (most frequent first)
    frequent_tokens_with_count = sorted(
        token_stats.items(), key=lambda x: x[1], reverse=True)
    frequent_tokens = [token[0] for token in frequent_tokens_with_count]

    return frequent_tokens

frequent_tokens = get_token_stats(text)
print(frequent_tokens)


# #### Implement the lowercase functionality to the get_token_stats function to avoid including the same word multiple times
def get_token_stats(raw_documents, lowercase=True):
    """
    :param raw_documents: iterable over raw text documents
    :param lowercase: boolean, default=True
        Convert all characters to lowercase before tokenizing

    :return token_stats: dict of {token:count}
    :return frequent_tokens: list tokens sorted by frequency
    """

    # add a check for the lowercase parmeter
    # use the same functionality as in fit
    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    token_stats = defaultdict(int)
    for doc in raw_documents:
        for term in doc.split(' '):
            token_stats[term] += 1

    # create a list of the tokens sorted by occurance (most frequent first)
    frequent_tokens_with_count = sorted(
        token_stats.items(), key=lambda x: x[1], reverse=True)
    frequent_tokens = [token[0] for token in frequent_tokens_with_count]

    return frequent_tokens

frequent_tokens = get_token_stats(text)
print(frequent_tokens)


# ### Complete max features paramater in fit, using get_token_stats
def fit(raw_documents, lowercase=True, max_features=None, frequent_tokens=None):
    """
    :param raw_documents: iterable over raw text documents
    :param max_features: int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus
    :param frequent_tokens: tokens ordered by frequency in the raw_documents
        Tokens with the same frequency are in a random order, accordingly
        tokens that display only once are not ordered alphabetically

    :return sorted_tokens: tokens sorted by frequency
    """

    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')
    distinct_tokens = set(all_tokens)

    # add max_features parameter check
    if max_features:
        # use the sorted list of tokens; filter to the # of max_features
        tokens_to_keep = frequent_tokens[0: max_features]
        # only keep tokens that are in the tokens_to_keep list
        distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, max_features=5, frequent_tokens=frequent_tokens)
print(sorted_tokens)


# #### Unintended results from passing the same data to related functions
# We are now begining to pass the same data from one function to another. Here we create the frequent_tokens variable in get_token_stats, then we pass it to the fit function.
# Both functions work on the same raw_documents; thus, if we preprocess the documents differently, we may get unintended results.
# As an example of this, change the lowercase parameter value to 'False'

sorted_tokens = fit(text, lowercase=False, max_features=5, frequent_tokens=frequent_tokens)
print(sorted_tokens)

# We no longer get five max_features when lowercase is 'False'
# We converted all documents to lowercase in get_token_stats, but have not done the same yet in fit. Therefore, even though we are asking for five max_features only four are returned. In this case, the missing term is 'This' which is always uppercase in the original documents (e.g. 'This is the first document')
# Some of the consequences are that we must recreate the same code in multiple places (add a lowercase check in both get_token_stats and fit) and we must remember all the parameters we used previously to avoid analyzing different text in each sequence of functions.
# Soon we will observe how classes alleviate issues like this
# ### Complete fit

# put all the parameters together

def fit(raw_documents, lowercase=True, stop_words=None, max_features=None, frequent_tokens=None):
    """
    :param raw_documents: iterable over raw text documents
    :param lowercase: boolean, default=True
        Convert all characters to lowercase before tokenizing
    :param stop_words: string {'english'} or list
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
    :param max_features: int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus
    :param frequent_tokens: tokens ordered by frequency in the raw_documents
        Tokens with the same frequency are in a random order, accordingly
        tokens that display only once are not ordered alphabetically

    :return sorted_tokens: tokens sorted by frequency
    """

    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')

    if stop_words == 'english':
        stop_words = ['a','of']

    if stop_words:
        distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        # remove stops words from frequent_tokens
        frequent_tokens = [token for token in frequent_tokens if token not in stop_words]
    else:
        distinct_tokens = set(all_tokens)

    if max_features:
        tokens_to_keep = frequent_tokens[0: max_features]
        distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(
    text, stop_words=['is','first'], max_features=5, frequent_tokens=frequent_tokens)
print(sorted_tokens)


# ### Implement Transform

# Transform in scikit-learn returns a sparse matrix. The matrix includes the indices and count of each token in the raw documents. We have already created this in get_token_stats.
#
# Let's update get_token_stats to avoid recalculating the token counts in transform.

def get_token_stats(raw_documents, lowercase=True):
    """
    :param raw_documents: iterable over raw text documents
    :param lowercase: boolean, default=True
        Convert all characters to lowercase before tokenizing

    :return token_stats: dict of {token:count}
    :return frequent_tokens: list tokens sorted by frequency
    """

    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    token_stats = defaultdict(int)
    for doc in raw_documents:
        for term in doc.split(' '):
            token_stats[term] += 1

    frequent_tokens_with_count = sorted(
        token_stats.items(), key=lambda x: x[1], reverse=True)
    frequent_tokens = [token[0] for token in frequent_tokens_with_count]

    # return both the token_stats and frequent_tokens
    return token_stats, frequent_tokens

# update tuple unpacking of token_stats and frequent_tokens
token_stats, frequent_tokens = get_token_stats(text)
print(token_stats)


def transform(raw_documents, sorted_tokens, token_stats):
    """
    :param raw_documents: iterable over raw text documents
    :param sorted_tokens: tokens sorted by frequency
    :param token_stats: tuple of (token:count)

    :return sparse_matrix: list of tuples with replicating a sparse matrix
        indicates the index of non-zero tokens in a dense matrix: (row_num, column_num, count)
    """

    # set a container for the sparse matrix output
    # expected output: (row_num, column_num, count)
    sparse_matrix = []

    # create a dict of the key:value pairs of the
    # {token:index} for each token
    # each token is a column in a dense matrix
    tokens_col_index = {token:ind for ind, token in enumerate(sorted_tokens)}

    # enumerate each raw_document to get the row number
    # (index of document in list)
    for row_num, doc in enumerate(raw_documents):
        # iterate through all tokens in the doc
        for token in doc.split(' '):
            # only retain selected tokens from fit
            if token in sorted_tokens:
                # (row_num, column_num, count)
                sparse_matrix.append((row_num, tokens_col_index[token], token_stats[token]))

    return sparse_matrix

sparse_matrix = transform(text, sorted_tokens=sorted_tokens, token_stats=token_stats)
print(sparse_matrix)


# #### Now that we have the implemented fit and transform, we run the entire process
token_stats, frequent_tokens = get_token_stats(raw_documents=text)

sorted_tokens = fit(
      raw_documents=text
    , lowercase=True
    , stop_words=None
    , max_features=5
    , frequent_tokens=frequent_tokens)

sparse_matrix = transform(
      raw_documents=text
    , sorted_tokens=columns
    , token_stats=token_stats)

print('COLUMNS: \n{} \n'.format(sorted_tokens))
print('DATA: \n{} \n'.format(sparse_matrix))
print('TOKEN_STATS: \n{} \n'.format(token_stats))
print('frequent_tokens: \n{}'.format(frequent_tokens))


# ### Inconveniences in the above approaches
# - The various functions are spread throughout the codebase
# - We have to keep giving new names to intermediary steps if we want to keep multiple versions (e.g. token_stats1, token_stats2)
# - If we run the code multiple times, it is difficult to remember the parameters we used
# - It is annoying to keep passing the same params into the multiple functions
# - We have to remember to run helper functions (e.g. get_token_stats) before we can run other functions, even though these should always occur
# - Multiple objects are returned from some functions, even though we may not always need both. We either have to call these functions (e.g get_token_stats) inside of multiple other functions (leading to duplicate computation) or bring them outside of the functions as be did above, complicating the code and usage.
#

# # Classes

# In the above code, we had a collection of related data and functions. A class acts as a container for these related data and functions.
#
# Think of a class as a blueprint that defines how to create objects. Classes use the following terminology:
# - Class: blueprint
# - Instance: a single object created from a class
# - Attributes: variables in a class
# - Methods: functions in a class

# In[57]:


# A simple class with one attribute

class CountVectorizer:

    def __init__(self, lowercase=True):
        self.lowercase = lowercase


# - Classes start with the word class
# - Class names use CamelCase
# - functions inside of a class (def) are called methods
# - variables inside of a class are called attributes (lowercase)
# - \__init\__ stands for initialization
# - the double underscores indicate that \__init\__ is a special or dunder (double underscore) method

# ### self is how we refer to an instane of a class

# - Classes build an object that retains the data with which it is initialized.
# - Unlike a function, we can recall the parameters used to fit the class
cv1 = CountVectorizer()
cv2 = CountVectorizer(lowercase=False)
print('self.lowercase for csv1: {}'.format(cv1.lowercase))
print('self.lowercase for csv2: {}'.format(cv2.lowercase))


# ### Classes combine related data and functions

# Review the fit function and add it to our CountVectorizer class
def fit(raw_documents):
    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')
    distinct_tokens = set(all_tokens)
    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text)
print(sorted_tokens)

class CountVectorizer:

    def __init__(self):
        self.vocabulary_ = None

    def fit(self, raw_documents):
        """
        :param raw_documents: iterable over raw text documents
        """

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')
        distinct_tokens = set(all_tokens)
        sorted_tokens = sorted(list(distinct_tokens))

        self.vocabulary_ = sorted_tokens

        return self

cv = CountVectorizer()
cv.fit(text)
print(cv.vocabulary_)


# When we place the fit function inside of our CountVectorizer Class, we call it a method. The code for the fit is the same except that we no longer return the vocabulary, instead we store it as a permanent attribute in the class called vocabulary\_.   We start by initializing vocabulary_ to None then populate it in the fit method. We will explore the impact of returning self later.
#
# - To use our class, we first create an instance of the class: cv = CountVectorizer()
# - Then we fit it with a corpora of documents: cv.fit(text)
# - By storing the distinct tokens (sorted\_tokens from the previous function), we can call the tokens at any later point: cv.vocabulary_
#
# NOTE: Here we are using common syntax in scikit-learn where an underscore after an attribute name (e.g. vocabulary_) means the attribute only has data after running a method.

cv1 = CountVectorizer()
cv2 = CountVectorizer()

cv1.fit(raw_documents=['just one sentence'])
print('self.vocabulary_ for csv1: {}'.format(cv1.vocabulary_))

cv2.fit(['a different sentence'])
print('self.vocabulary_ for csv2:  {}'.format(cv2.vocabulary_))


# We can create multiple instances of the same class and each will remember there separate attributes. This way we only need to provide a different name for the overall instance, not every attribute if we want to store multiple CountVectorizer instances with different parameters

# #### add lowercase

def fit(raw_documents, lowercase=False):
    """
    :param raw_documents: iterable over raw text documents
    :param lowercase: boolean, default=True
        Convert all characters to lowercase before tokenizing

    :return sorted_tokens: tokens sorted by frequency
    """

    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')
    distinct_tokens = set(all_tokens)
    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, lowercase=True)
print(sorted_tokens)


class CountVectorizer:

    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.vocabulary_ = None

    def fit(self, raw_documents):
        """
        :param raw_documents: iterable over raw text documents
        """

        # add a check for the lowercase parameter
        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')
        distinct_tokens = set(all_tokens)
        sorted_tokens = sorted(list(distinct_tokens))

        self.vocabulary_ = sorted_tokens

        return self

cv = CountVectorizer()
cv.fit(text)
print(cv.vocabulary_)


# By adding self to the method signature, we can use all of the stored attributes in the class instance (i.e. in the \__init\__) without having to pass them in as parameters to the method. For instance, we use self.lowercase in the fit method, only by passing in self.

# #### Add stop words

# add in a stop words parameter
def fit(raw_documents, stop_words=None):
    """
    :param raw_documents: iterable over raw text documents
    :param stop_words: string {'english'} or list
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    :return sorted_tokens: tokens sorted by frequency
    """

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')

    # add a check if you add your own stop words
    if stop_words == 'english':
        # include a default list of stop words in the function
        stop_words = ['a','of']

    if stop_words:
        distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
    else:
        distinct_tokens = set(all_tokens)

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, stop_words=None)
print(sorted_tokens)


class CountVectorizer:
    # set a static list of stop words as part of the class
    ENGLISH_STOP_WORDS = ['a', 'of', 'in', 'the', 'to']

    def __init__(self, lowercase=True, stop_words=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.vocabulary_ = None

    def fit(self, raw_documents):
        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')

        # add stop words check
        stop_words = self.stop_words
        if stop_words == 'english':
            stop_words = CountVectorizer.ENGLISH_STOP_WORDS

        # keep all terms if stop_words is None
        if stop_words:
            distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        else:
            distinct_tokens = set(all_tokens)

        sorted_tokens = sorted(list(distinct_tokens))
        self.vocabulary_ = sorted_tokens

        return self

cv = CountVectorizer(stop_words=['a', 'of'])
cv.fit(text)
print(cv.vocabulary_)


# Again, we set the attributes of CountVectorizer a single time when instantiating CountVectorizer. We then can reuse all of the attributes on self. In addition to using self.lower_case, we are also using self.stop_words.
#
# We also want to include a static list of stop words that will be the same for all instances of CountVectorizer. In this case, we do not include these stop words (i.e. ENGLISH_STOP_WORDS) as part of \__init\__, since this is usually for instance specific attributes. Rather, we set it as part of the class, only indented in Countvectorizer, but not inside a method.
#
# To call ENGLISH_STOP_WORDS, we can use the class without ever creating an instance of the class

CountVectorizer.ENGLISH_STOP_WORDS

# use ENGLISH_STOP_WORDS by setting stop_words = 'english'
cv = CountVectorizer(stop_words='english')
cv.fit(text)
print(cv.vocabulary_)

# change ENGLISH_STOP_WORDS
CountVectorizer.ENGLISH_STOP_WORDS = ['again','and']

# since ENGLISH_STOP_WORDS changes the entire class, previous instances are impacted
cv.fit(text)
print(cv.vocabulary_)


# We can still add our own list of stop_words, ignoring ENGLISH_STOP_WORDS entirely
cv = CountVectorizer(stop_words=['document','first'])
cv.fit(text)
print(cv.vocabulary_)


# #### add get_token_stats
def get_token_stats(raw_documents, lowercase=True):
    if lowercase:
        raw_documents = [doc.lower() for doc in raw_documents]

    token_stats = defaultdict(int)
    for doc in raw_documents:
        for term in doc.split(' '):
            token_stats[term] += 1

    sorted_tokens_with_count = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)
    sorted_tokens = [token[0] for token in sorted_tokens_with_count]

    return token_stats, sorted_tokens

token_stats, frequent_tokens = get_token_stats(text)
print(token_stats)

def fit(raw_documents, max_features=None, frequent_tokens=None):
    """
    :param raw_documents: iterable over raw text documents
    :param max_features: int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus
    :param frequent_tokens: tokens ordered by frequency in the raw_documents
        Tokens with the same frequency are in a random order, accordingly
        tokens that display only once are not ordered alphabetically

    :return sorted_tokens: tokens sorted by frequency
    """

    combined_sentences = ' '.join(raw_documents)
    all_tokens = combined_sentences.split(' ')
    distinct_tokens = set(all_tokens)

    if max_features:
        tokens_to_keep = frequent_tokens[0: max_features]
        distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

    sorted_tokens = sorted(list(distinct_tokens))

    return sorted_tokens

sorted_tokens = fit(text, max_features=5, frequent_tokens=frequent_tokens)
print(sorted_tokens)


class CountVectorizer:
    ENGLISH_STOP_WORDS = ['a', 'of', 'in', 'the', 'to']

    def __init__(self, lowercase=True, stop_words=None, max_features=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_features = max_features
        self.vocabulary_ = None
        self.token_stats_ = None
        self.frequent_tokens_ = None

    def fit(self, raw_documents):
        stop_words = self.stop_words
        max_features = self.max_features

        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')

        # add stop words check
        if stop_words == 'english':
            stop_words = CountVectorizer.ENGLISH_STOP_WORDS

        if stop_words:
            distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        else:
            distinct_tokens = set(all_tokens)

        if self.frequent_tokens_ is None:
            self._get_token_stats(raw_documents)

        if max_features:
            tokens_to_keep = self.frequent_tokens_[0: max_features]
            distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

        sorted_tokens = sorted(list(distinct_tokens))
        self.vocabulary_ = sorted_tokens

        return self

    def _get_token_stats(self, raw_documents):
        token_stats = defaultdict(int)
        for doc in raw_documents:
            for term in doc.split(' '):
                token_stats[term] += 1

        frequent_tokens_with_count = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)
        frequent_tokens = [token[0] for token in frequent_tokens_with_count]

        self.token_stats_ = token_stats
        self.frequent_tokens_ = frequent_tokens

        return self

cv = CountVectorizer(stop_words=['a', 'of'])
cv.fit(text)
print(cv.vocabulary_)


# #### add transform

def transform(raw_documents, sorted_tokens, token_stats):
    """
    :param raw_documents: iterable over raw text documents
    :param sorted_tokens: tokens sorted by frequency
    :param token_stats: tuple of (token:count)

    :return sparse_matrix: list of tuples with replicating a sparse matrix
        indicates the index of non-zero tokens in a dense matrix: (row_num, column_num, count)
    """

    sparse_matrix = []
    tokens_col_index = {token:ind for ind, token in enumerate(sorted_tokens)}

    for row_num, doc in enumerate(raw_documents):
        for token in doc.split(' '):
            if token in sorted_tokens:
                sparse_matrix.append((row_num, tokens_col_index[token], token_stats[token]))

    return sparse_matrix

sparse_matrix = transform(text, sorted_tokens=sorted_tokens, token_stats=token_stats)
print(sparse_matrix)


class CountVectorizer:
    ENGLISH_STOP_WORDS = ['a', 'of', 'in', 'the', 'to']

    def __init__(self, lowercase=True, stop_words=None, max_features=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_features = max_features
        self.vocabulary_ = None
        self.token_stats_ = None
        self.frequent_tokens_ = None

    def fit(self, raw_documents):
        stop_words = self.stop_words
        max_features = self.max_features

        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')

        # add stop words check
        if stop_words == 'english':
            stop_words = CountVectorizer.ENGLISH_STOP_WORDS

        if stop_words:
            distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        else:
            distinct_tokens = set(all_tokens)


        if self.frequent_tokens_ is None:
            self._get_token_stats(raw_documents)

        if max_features:
            tokens_to_keep = self.frequent_tokens_[0: max_features]
            distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

        sorted_tokens = sorted(list(distinct_tokens))
        self.vocabulary_ = sorted_tokens

        return self

    def _get_token_stats(self, raw_documents):
        token_stats = defaultdict(int)
        for doc in raw_documents:
            for term in doc.split(' '):
                token_stats[term] += 1

        frequent_tokens_with_count = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)
        frequent_tokens = [token[0] for token in frequent_tokens_with_count]

        self.token_stats_ = token_stats
        self.frequent_tokens_ = frequent_tokens

        return self

    def transform(self, raw_documents):
        """
        :param raw_documents: iterable over raw text documents
        :param sorted_tokens: tokens sorted by frequency
        :param token_stats: tuple of (token:count)

        :return sparse_matrix: list of tuples with replicating a sparse matrix
            indicates the index of non-zero tokens in a dense matrix: (row_num, column_num, count)
        """

        if self.vocabulary_ is None:
            raise('Must run fit before transform')

        sorted_tokens = self.vocabulary_
        token_stats = self.token_stats_

        sparse_matrix = []
        tokens_col_index = {token:ind for ind, token in enumerate(sorted_tokens)}
        for row_num, doc in enumerate(raw_documents):
            for token in doc.split(' '):
                if token in sorted_tokens:
                    sparse_matrix.append((row_num, tokens_col_index[token], token_stats[token]))

        return sparse_matrix


cv = CountVectorizer(stop_words='english')
cv.fit(text)
sparse_matrix = cv.transform(text)
print(sparse_matrix)


# #### Add get_feature_names()

class CountVectorizer:
    ENGLISH_STOP_WORDS = ['a', 'of', 'in', 'the', 'to']

    def __init__(self, lowercase=True, stop_words=None, max_features=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_features = max_features
        self.vocabulary_ = None
        self.token_stats_ = None
        self.frequent_tokens_ = None

    def fit(self, raw_documents):
        stop_words = self.stop_words
        max_features = self.max_features

        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')

        # add stop words check
        if stop_words == 'english':
            stop_words = CountVectorizer.ENGLISH_STOP_WORDS

        if stop_words:
            distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        else:
            distinct_tokens = set(all_tokens)


        if self.frequent_tokens_ is None:
            self._get_token_stats(raw_documents)

        if max_features:
            tokens_to_keep = self.frequent_tokens_[0: max_features]
            distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

        sorted_tokens = sorted(list(distinct_tokens))
        self.vocabulary_ = sorted_tokens

        return self

    def _get_token_stats(self, raw_documents):
        token_stats = defaultdict(int)
        for doc in raw_documents:
            for term in doc.split(' '):
                token_stats[term] += 1

        frequent_tokens_with_count = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)
        frequent_tokens = [token[0] for token in frequent_tokens_with_count]

        self.token_stats_ = token_stats
        self.frequent_tokens_ = frequent_tokens

        return self

    def transform(self, raw_documents):
        """
        :param raw_documents: iterable over raw text documents
        :param sorted_tokens: tokens sorted by frequency
        :param token_stats: tuple of (token:count)

        :return sparse_matrix: list of tuples with replicating a sparse matrix
            indicates the index of non-zero tokens in a dense matrix: (row_num, column_num, count)
        """

        if self.vocabulary_ is None:
            raise('Must run fit before transform')

        sorted_tokens = self.vocabulary_
        token_stats = self.token_stats_

        sparse_matrix = []
        tokens_col_index = {token:ind for ind, token in enumerate(sorted_tokens)}
        for row_num, doc in enumerate(raw_documents):
            for token in doc.split(' '):
                if token in sorted_tokens:
                    sparse_matrix.append((row_num, tokens_col_index[token], token_stats[token]))

        return sparse_matrix

    def get_feature_names(self):
        """ Get an alphabetical list of the vocabulary learned in the fit method """

        return [token for token in sorted(self.vocabulary_)]

cv = CountVectorizer(stop_words='english')
cv.fit(text)
print(cv.get_feature_names())


# NOTE: the below code will place greater focus on learning Python classes than exactly replicating the CountVectorizer codebase. Thus, the code will have substantial differences in places. The primary differences are do to the following reasons:
# - Error handling - To reduce user errors, scikit-learn has includes substantial error handling code
# - Optimization - scikit-learn optimizes the code to reduce memory usage and reduce necessary calculations
# - Compatibility - scikit-learn makes sure that the codebase is compatible with multiple versions of python
# - Inheritance - other classes (e.g. TFIDFVectorizer) reuse much of the code of CountVectorizer, so scikit-learn uses inheritance (we will not cover that here) to reduce code and encourage code resue.

# # EXTRA MATERIAL

# ### \__repr\__

# In the fit method, we have used 'return self'. This returns a representation of the class instance, which we can change with the \__repr\__ method.
#
# Using \__repr\__, we can print out our own string to explain the class instance; this will help us replicate how the scikit-learn CountVectorizer printed out all the set parameters for the CountVectorizer instance

cv = CountVectorizer(stop_words='english')
cv.fit(text)

# Let review a simplified example

class CountVectorizer:

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

cv = CountVectorizer()
cv


# By default, the class instance prints the name of the class and memory related information. We will update this to say something more meaningful
# Now let's add a __repr__ method to change the output when we print the class instance

class CountVectorizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    # add a __repr__
    def __repr__(self):
        return "CountVectorizer(lowercase={})".format(self.lowercase)

cv = CountVectorizer()
cv


# Now let's add a fit method without 'return self'

class CountVectorizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    # add a fit method
    def fit(self, raw_documents):
        # simplified fit
        self.vocabulary_ = raw_documents

    def __repr__(self):
        return "CountVectorizer(lowercase={})".format(self.lowercase)

cv = CountVectorizer()
cv.fit(text)


# When we run the cv.fit(text) no output displays. This is because the method does not include a return value or print statements
# Now let's add a fit method without 'return self'

class CountVectorizer:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    # add a fit method
    def fit(self, raw_documents):
        self.vocabulary_ = raw_documents

        # add return self
        return self

    def __repr__(self):
        return "CountVectorizer(lowercase={})".format(self.lowercase)

cv = CountVectorizer()
cv.fit(text)


# Once we add 'return self' to the fit method, we print out the \__repr\__ value by default

# put everything together with a complete __repr__ method (found at the end of the class)

class CountVectorizer:
    ENGLISH_STOP_WORDS = ['a', 'of', 'in', 'the', 'to']

    def __init__(self, lowercase=True, stop_words=None, max_features=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_features = max_features
        self.vocabulary_ = None
        self.token_stats_ = None
        self.frequent_tokens_ = None

    def fit(self, raw_documents):
        stop_words = self.stop_words
        max_features = self.max_features

        if self.lowercase:
            raw_documents = [doc.lower() for doc in raw_documents]

        combined_sentences = ' '.join(raw_documents)
        all_tokens = combined_sentences.split(' ')

        if stop_words == 'english':
            stop_words = CountVectorizer.ENGLISH_STOP_WORDS

        if stop_words:
            distinct_tokens = [token for token in set(all_tokens) if token not in stop_words]
        else:
            distinct_tokens = set(all_tokens)


        if self.frequent_tokens_ is None:
            self._get_token_stats(raw_documents)

        if max_features:
            tokens_to_keep = self.frequent_tokens_[0: max_features]
            distinct_tokens = [token for token in distinct_tokens if token in tokens_to_keep]

        sorted_tokens = sorted(list(distinct_tokens))
        self.vocabulary_ = sorted_tokens

        return self

    def _get_token_stats(self, raw_documents):
        token_stats = defaultdict(int)
        for doc in raw_documents:
            for term in doc.split(' '):
                token_stats[term] += 1

        frequent_tokens_with_count = sorted(token_stats.items(), key=lambda x: x[1], reverse=True)
        frequent_tokens = [token[0] for token in frequent_tokens_with_count]

        self.token_stats_ = token_stats
        self.frequent_tokens_ = frequent_tokens

        return self

    def transform(self, raw_documents):
        """
        :param raw_documents: iterable over raw text documents
        :param sorted_tokens: tokens sorted by frequency
        :param token_stats: tuple of (token:count)

        :return sparse_matrix: list of tuples with replicating a sparse matrix
            indicates the index of non-zero tokens in a dense matrix: (row_num, column_num, count)
        """

        if self.vocabulary_ is None:
            raise('Must run fit before transform')

        sorted_tokens = self.vocabulary_
        token_stats = self.token_stats_

        sparse_matrix = []
        tokens_col_index = {token:ind for ind, token in enumerate(sorted_tokens)}
        for row_num, doc in enumerate(raw_documents):
            for token in doc.split(' '):
                if token in sorted_tokens:
                    sparse_matrix.append((row_num, tokens_col_index[token], token_stats[token]))

        return sparse_matrix

    def __repr__(self):
        return "CountVectorizer(lowercase={}, max_features={}, stop_words={})".format(
            self.lowercase, self.max_features, self.stop_words)

cv = CountVectorizer(stop_words='english')
cv.fit(text)


# #### Error Handling
class CountVectorizer:

    def __init__(self, lowercase=True, stop_words=None, max_features=None):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.max_features = max_features

        if max_features is not None:
            if (not isinstance(max_features, int)) or (max_features <= 0):
                raise ValueError("max_features={}, neither a positive integer nor None".format(max_features))

cv = CountVectorizer(max_features=1.5)
