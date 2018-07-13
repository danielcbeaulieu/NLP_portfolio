
# coding: utf-8

# ## Text Vectorization (count-based methods) Solution
#
# ##### Author: Daniel Beaulieu | danbeaulieu@gmail.com.com

# In[1]:


from IPython.core.display import display, HTML
from IPython.display import Image
from gensim.summarization.bm25 import get_bm25_weights
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# - **tokenizing** strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
# - **counting** the occurrences of tokens in each document.
# - **normalizing** and weighting with diminishing importance tokens that occur in the majority of samples / documents.
# create a list of documents
text = [  'This is the first document'
        , 'This is the second second document'
        , 'And the third one'
        , 'Is it the first document again']


# ### Step 1 - import from sklearn
from sklearn.feature_extraction.text import CountVectorizer

# ### Step 2 - instantiate
# create an instance of countvectorizer
vect = CountVectorizer()

# when we print vect, we see its hyperparameters
print(vect)

# ### Step 3 - fit
# The vectorizer learns the vocabulary when we fit it with our documents.
vect.fit(text)

print('ORIGINAL_SENTENCES: \n {} \n'.format(text))
print('FEATURE_NAMES: \n {}'.format(vect.get_feature_names()))

# ### Step 4 - transform
# Transform creates a sparse matrix, identifying the indices where terms are stores in each document

vect.transform(text)

# ### Sparsity
# As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have many feature values that are zeros (typically more than 99% of them).
# For instance a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.

print(vect.transform(text))

# ### Sparse Matrix
# **Compressed Sparse Row (CSR)** format stores the non-zero entries of a sparse matrix.
# This is easier to understand when we covert the sparse matrix into a dense matrix or pandas DataFrame

vect.transform(text).toarray()

import pandas as pd

# store the dense matrix
data = vect.transform(text).toarray()

# store the learned vocabulary
columns = vect.get_feature_names()

# combine the data and columns into a dataframe
pd.DataFrame(data, columns=columns)


# ### Bag of Words
# We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors.
# Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
# Use the trained CountVectorizer to vectorize the  sentences. Create a dataframe with the dense results.

example_text = ['again we observe a document'
               , 'the second time we have see this text']

data = vect.transform(example_text).toarray()
columns = vect.get_feature_names()

pd.DataFrame(data, columns=columns)

# ### fit_transform
# - we can combine the training and transformation into a single method. This is a common process in the sklearn api, as we often want to learn something from a training data set and apply the results to testing or production data
vect = CountVectorizer()
vect.fit_transform(text)

## Customize the Transformer
# During the process of vectorizing the text, we can apply numerous transformations to modify the text and resulting vectors.
# ### lowercase
# - boolean, True by default
# - Convert all characters to lowercase before tokenizing.

# by instantiating CountVectorizer with differnt parameters,
# we can change the vocabulary lowercase determines if all words
# should be lowercase, setting it to False includes uppercase words

vect = CountVectorizer(lowercase=False)
vect.fit(text)
print(vect.get_feature_names())

# ### stop_words
# - string {‘english’}, list, or None (default)
#  - If None, no stop words will be used.
#  - If ‘english’, a built-in stop word list for English is used.
#  - If list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
# - max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words

# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents
vect = CountVectorizer(stop_words='english')
vect.fit(text)
print(vect.get_feature_names())

# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents
vect = CountVectorizer(stop_words=['first','second','third'])
vect.fit(text)
print(vect.get_feature_names())

# ### vocabulary
# - Mapping or iterable, optional
# - Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If not given, a vocabulary is determined from the input documents.

# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents
vect = CountVectorizer(vocabulary=['first','second','third'])
vect.fit(text)
print(vect.get_feature_names())

# ### max_features
# - int or None, default=None
# - If not None, build a vocabulary that only consider the top  max_features ordered by term frequency across the corpus.

vect = CountVectorizer(max_features=5)
vect.fit(text)
print(vect.get_feature_names())

# ### max_df
# - float in range [0.0, 1.0] or int, default=1.0
# - When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

vect = CountVectorizer(max_df=.5)
vect.fit(text)
print(vect.get_feature_names())

# ### min_df
# - float in range [0.0, 1.0] or int, default=1
# - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

vect = CountVectorizer(min_df=.5)
vect.fit(text)
print(vect.get_feature_names())

# ### ngram_range
# - tuple (min_n, max_n)
# - The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

# max features determines the maximum number of features to display
vect = CountVectorizer(ngram_range=(2,2), max_features=5)
vect.fit(text)
print(vect.get_feature_names())

# ### binary
# - boolean, default=False
# - If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.

# max features determines the maximum number of features to display
vect = CountVectorizer(binary=True)
vect.fit_transform(['Two Two different words words']).toarray()

# ### analyzer
# - String, {‘word’, ‘char’, ‘char_wb’} or callable
# - Specifies whether to use n_grams of words or characters
# - Character n_grams are useful in certain content, such as genomics with DNA sequences (e.g. GCTATCAFF...)

# max features determines the maximum number of features to display
vect = CountVectorizer(analyzer='char', ngram_range=(2,2))
vect.fit(text)
print(vect.get_feature_names())

# ### Limitations of the Bag of Words representation
#  collection of bigrams (n=2), where occurrences of pairs of consecutive words are counted.
# alternatively consider a collection of character n-grams, a representation resilient against misspellings and derivations.
# For example, let’s say we’re dealing with a corpus of two documents: ['words', 'wprds']. The second document contains a misspelling of the word ‘words’. A simple bag of words representation would consider these two as very distinct documents, differing in both of the two possible features. A character 2-gram representation, however, would find the documents matching in 4 out of 8 features, which may help the preferred classifier decide better:

# # Attributes
# In scikit-learn attributes are often provided to store information of the instance of the transformer or model.
# Many attributes are only available after the model is fit. For instance the learned vocabulary does not exist in Countvectorizer until text data has been provided with the fit method. Until the data is provided these attributes do not exist. The notation for these learned attributes is a trailing underscore after the attribute name (e.g. vocabulary_).

vect = CountVectorizer(max_features=5)
vect.fit(text)
print(vect.get_feature_names())

# ### vocabulary_
# - dict
# - A mapping of terms to feature indices.

vect.vocabulary_

# ### stop\__words_\_
# - set
# - Terms that were ignored because they either:
#  - occurred in too many documents (max_df)
#  - occurred in too few documents (min_df)
#  - were cut off by feature selection (max_features)

vect.stop_words_

# ### Term-Frequency Problems
#
# "The **main problem with the term-frequency approach is that it scales up frequent terms and scales down rare terms which are empirically more informative than the high frequency terms.**
# The basic intuition is that a term that occurs frequently in many documents is not a good discriminator; the important question here is: why would you, in a classification problem for instance, emphasize a term which is almost present in the entire corpus of your documents ?
#
# The tf-idf weight comes to solve this problem. **What tf-idf gives is how important is a word to a document**
# in a collection, and that’s why tf-idf incorporates local and global parameters, because it takes in consideration not only the isolated term but also the term within the document collection. **What tf-idf then does to solve that problem, is to scale down the frequent terms while scaling up the rare terms; a term that occurs 10 times more than another isn’t 10 times more important than it, that’s why tf-idf uses the logarithmic scale to do that."**

# ### TFIDF
#
# In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.
#
# In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.
# Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency:
#
# - tf-idf(t,d) = tf(t,d) * idf(t)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

text

tfidf_vect = TfidfVectorizer()
pd.DataFrame(tfidf_vect.fit_transform(text).toarray(), columns=columns)

# ### TFIDF Analysis
# As we look at the tfidf score (which have a range of 0-1), high score occur for words that show up frequently in specific sentence but infrequenty overall. Low score occur in words that show up frequenty across all documents.
#
# - **'Second' has a high score** as it shows up twice in document two and not in any other documents
# - **'The' has a low score** as it show up in all documents

# ## TFIDF Calculation

# #### term frequency (tf)
#
# How often does each term exist in each document.
#
# Term frequency is the numerator; thus, the tfidf score for a term increases in documents where it is frequent.
vect = CountVectorizer()
tf = vect.fit_transform(text).toarray()
print(tf)


# #### inverse document frequency (idf)
# Calculation: log(\# document in the corpus / # documents where the term appears)
# - **The # of documents in the corpus has no effect** as it is the same for all terms
# - **As the # of documents in which the term appears increases, the idf decreases**; thus terms that show up in many different documents (e.g. stop words) recieve low tfidf scores as they are not important terms to define the meaning of the document
# - As a sub-linear function, we take the **log because the relevance does not increase proportionally with the term frequency**. As an example if a term shows up in 1M docs or in 2M docs, the effect is not the same as if it has shown up in 1 doc or 2 docs times respectively. In other words there is a relative threshold.

# idf calculation
print( np.log(len(tf) / tf.sum(axis=0)) )

# when we use sum(axis=0) we take the sum of each column
# as opposed to a scalar sum (single # result) of all values
tf.sum(axis=0)


# #### scikit-learn calculation modifications
# scikit-learn further modifies the caluclation for adding one to the numerator, denominator, and log to avoid divide by zero errors
idf = np.log( (len(tf)+1) / (tf.sum(axis=0)+1) ) + 1
print(idf)

# value as stored from sklearn in tfidf_vect
print(tfidf_vect.idf_)

# #### term frequency * inverse document frequency (tf*idf)

tfidf = pd.DataFrame(tf*idf)
tfidf

# #### term vector normalization
# The use of the simple tfidf does not account for the length of the document. Additionally it provides opportunities for spammers to repeat the term many times to make it seem more important.
# To solve these issues, we normalize each vector. By default TfidfVectorizer uses an 'l2' normalization.

# tf*idf is equivalent to using TfidfVectorizer without a norm
tfidf_vect = TfidfVectorizer(norm=None)
pd.DataFrame(tfidf_vect.fit_transform(text).toarray())

from sklearn.preprocessing import normalize

pd.DataFrame(normalize(tfidf, norm='l2'))

# ### BM25
# BM25 is often a better algorithm than tfidf to determine term importance as it takes that document length into account.
# The Probabilistic Relevance Framework - BM25 and Beyond: http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf

# ### Co-occurrence matrix
#
# "Similar words tend to occur together and will have similar context for example
#
# **Co-occurrence** – For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window.
#
# **Context Window** – Context window is specified by a number and the direction.
#
# Let’s say there are V unique words in the corpus. So Vocabulary size = V. The columns of the Co-occurrence matrix form the context words. The different variations of Co-Occurrence Matrix are-
#
# A co-occurrence matrix of size V X V. Now, for even a decent corpus V gets very large and difficult to handle. So generally, this architecture is never preferred in practice.
# A co-occurrence matrix of size V X N where N is a subset of V and can be obtained by removing irrelevant words like stopwords etc. for example. This is still very large and presents computational difficulties.

# Co-occurance matrix
# review text
text

vect = CountVectorizer(ngram_range=(1,1))
X = vect.fit_transform(text)
cooccurance = (X.T * X) # this is co-occurrence matrix in sparse csr format
cooccurance.setdiag(0) # fill same word cooccurence to 0

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

terms = vect.get_feature_names()
cooccur_df = pd.DataFrame(cooccurance.todense(), columns=terms, index=terms)
sns.heatmap(cooccur_df)
