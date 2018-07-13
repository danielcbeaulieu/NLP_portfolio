
# coding: utf-8

# # Phrase (collocation) Detection Solution
#
# ###### Author: Daniel Beaulieu | danbeaulieu@gmail.com

# 1. Acronym replacement
# 2. SpaCy POS phrases
# 3. Gensim Phrases and Phraser

import spacy
import pandas as pd
from sqlalchemy import create_engine
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
from IPython.core.display import display, HTML
from configparser import ConfigParser, ExtendedInterpolation

# configuration for data, acronyms, and gensim paths
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../../config.ini')

DB_PATH = config['DATABASES']['PROJECT_DB_PATH']
AIRLINE_ACRONYMS_FILEPATH = config['NLP']['AIRLINE_ACRONYMS_FILEPATH']
AIRLINE_MATCHED_TEXT_PATH = config['NLP']['AIRLINE_MATCHED_TEXT_PATH']
AIRLINE_CLEANED_TEXT_PATH = config['NLP']['AIRLINE_CLEANED_TEXT_PATH']
GENSIM_DICTIONARY_PATH = config['NLP']['GENSIM_DICTIONARY_PATH']
GENSIM_CORPUS_PATH = config['NLP']['GENSIM_CORPUS_PATH']


# #### Load data on airline fees
engine = create_engine(DB_PATH)
df = pd.read_sql("SELECT * FROM Sections", con=engine)

# filter to relevant sections
df = df[df['section_text'].str.contains('fee')]
df.head()

# store section matches in list
text = [section for section in df['section_text'].values]

# review first sentence of a section match
text[0][0:299]


# ### SpaCy - Preprocessing

get_ipython().run_cell_magic('time', '', "\n# load spacy nlp model\n# use 'en' if you don't have the lg model\nnlp = spacy.load('en_core_web_lg')")


# ##### Text Preprocessing - Acronyms
#

# read csv with airline industry acronyms
airline_acronyms = pd.read_csv(AIRLINE_ACRONYMS_FILEPATH)
airline_acronyms.head()


# **Curate the acronyms:**
acronyms = {}

for ind, row in airline_acronyms.iterrows():
    # get the acronym and convert it to lowercase
    acronym = row['Acronym'].lower()

    # clean acronym definition:
    # lower case, strip excess space, replace spaces with underscores to create a single term
    definition = row['Definition'].lower().strip().replace(' ','_')

    # add acronyms/definitions pairs to the acronyms dict
    # ignore two character acronyms as they often match actual words
    # e.g. 'at' == 'air traffic'
    if len(acronym) > 2:
        acronyms[acronym] = definition

# view the first few acronyms
list(acronyms.items())[0:5]  # convert to list as dict is not subscriptable


# ##### Identify acronyms that exist in text
# WARNING SLOW!
get_ipython().run_cell_magic('time', '', "\nif 1 == 0:\n    # review the acronyms\n    acronym_matches = []\n\n    # create a nlp pipe to iterate through the text\n    for doc in nlp.pipe(text, disable=['tagger','ner']):    \n        # iterate through each word in the sentence\n        for token in doc:\n            token = token.text.lower()\n            # check if token is an acronym\n            # add matches (acronym and definition) to acronym_matches\n            if token in acronyms:\n                acronym_matches.append((token, acronyms[token]))\n\n    # review all matching acronyms      \n    for match in set(acronym_matches):\n        print(match)")


# update acronyms list to remove ambiguous acronyms
acronyms_to_remove = ['cat','app','grade','self','basic','did','far']
for term in acronyms_to_remove:
    acronyms.pop(term)


# ###### collect sentences about fees for phrase model
def collect_phrase_model_sents(matcher, doc, i, matches):
    # identify matching spans (phrases)
    match_id, start, end = matches[i]
    span = doc[start:end]

    # keep only words, lemmatize tokens, remove punctuation
    sent = [str(token.lemma_).lower()
            for token in span.sent if token.is_alpha]

    # replace acronyms
    sent = [acronyms[token] if token in acronyms else token
            for token in sent]

    # collect matching (cleaned) sents
    matched_sents.append(sent)


# ##### match sentences with the word fee or fees
#
# WARNING SLOW!

get_ipython().run_cell_magic('time', '', "\nif 1 == 0:\n    # match sentences with the word fee or fees\n    matched_sents = []\n    pattern = [[{'LOWER': 'fee'}], [{'LOWER': 'fees'}]]\n\n    matcher = Matcher(nlp.vocab)\n    \n    # use *patterns to add more than one pattern at once\n    matcher.add('fees', collect_phrase_model_sents, *pattern)\n\n    for doc in nlp.pipe(text, disable=['tagger','ner']):    \n        matcher(doc)")

print('Number of matches: {} \n'.format(len(matched_sents)))

print('Example Match:')
print(matched_sents[0])


# ##### Export matched text to avoid repeating processing
# uncomment below to write the matched text to a .txt file for later use

# with open(AIRLINE_MATCHED_TEXT_PATH, 'w') as f:
#    for line in matched_sents:
#        line = ' '.join(line) + '\n'
#        line = line.encode('ascii', errors='ignore').decode('ascii')
#        f.write(line)


AIRLINE_MATCHED_TEXT_PATH

# read matched text
with open(AIRLINE_MATCHED_TEXT_PATH, 'r') as f:
    matched_sents_full = [line for line in f.readlines()]
    matched_sents = [line.split() for line in matched_sents_full]

# store all matched sentences in a dataframe
matches_df = pd.DataFrame(matched_sents_full, columns=['sentences'])

# remove duplicates
matches_df = matches_df.drop_duplicates()

matches_df.head()


# ### Use SpaCy part of speech (POS) to create phrases
# combine the matched sentence tokens and parse it with SpaCy
text = ' '.join(matched_sents[0])
text

# ##### Determine which NLP components can be disabled

def view_pos(doc, n_tokens=5):
    """ print SpaCy POS information about each token in a provided document """
    print('{:15} | {:10} | {:10} | {:30}'.format('TOKEN','POS','DEP_','LEFTS'))
    for token in doc[0:n_tokens]:
        print('{:15} | {:10} | {:10} | {:30}'.format(
            token.text, token.head.pos_,token.dep_, str([t.text for t in token.lefts])))


# observe which part of speech (pos) attributes are disabled by named entity recognition (ner)
pos_doc = nlp(text, disable=['ner'])
view_pos(pos_doc)

# observe which part of speech (pos) attributes are disabled by parser
pos_doc = nlp(text, disable=['ner','parser'])
view_pos(pos_doc)

# observe which part of speech (pos) attributes are disabled by tagger
pos_doc = nlp(text, disable=['ner','tagger'])
view_pos(pos_doc, n_tokens=10)

# use explain to define any token.dep_ attributes
spacy.explain('dobj')

# ##### Extract phrases by identifying tokens describing an object

# add stop words to SpaCy
# this enables the .is_stop attribute with common stop words
from spacy.lang.en.stop_words import STOP_WORDS

for word in STOP_WORDS:
    lex = nlp.vocab[word]
    lex.is_stop = True

def create_pos_phrases(doc):

    phrases = []

    doc = nlp(doc, disable=['ner','tagger'])
    for token in doc:
        # find any objects (e.g. direct objects )
        if 'obj' in token.dep_:
            token_text = token.lemma_.lower()

            # find any dependent terms to the left of (preceeding) the object
            # ignore dependent terms that are not stop words
            for left_term in (t.text for t in token.lefts if t.is_stop is False):
                # combine the dependent term and object, separated by an underscore
                # e.g. travel agency ==> travel_agency
                phrase = '{}_{}'.format(left_term,token_text)
                phrases.append(phrase)

    # convert list of distinct phrases into a sentence
    return ' '.join(set(phrases))

print(create_pos_phrases(matched_sents_full[0]))

get_ipython().run_cell_magic('time', '', "\n# apply the custom function to every element in the dataframe\nmatches_df['pos_phrases'] = matches_df.sentences.apply(create_pos_phrases)")

matches_df.head()


# ##### Pandas Apply
# apply is an efficient and fast approach to 'apply' a function to every element in a row. applymap does the same to every element in the entire dataframe (e.g. convert all ints to floats)
# Example: https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_dataframes/

# create a small dataframe with example data
test_df = pd.DataFrame({'col1':range(0,3),'col2':range(3,6)})
test_df


# apply a built-in function to each element in a column
test_df['col1'].apply(float)

# apply a custom function to every element in a column
def add_five(row):
    return row + 5

test_df['col1'].apply(add_five)

# apply an annonomous function to every element in a column
test_df['col1'].apply(lambda x: x+5)


# apply a built-in function to every element in a dataframe
test_df.applymap(float)  # applymap


# ### Collocations
#
# Create a function that returns a window of size n over a given sentence.
# For the sentence **'rather than pay the fee'** return the following if the window is n=3:
# - ['rather', 'than', 'pay'],
# - ['than','pay','the']
# - ['pay', 'the','fee']
# - ...
#

# example sentence
sent = ' '.join(matches_df['sentences'][0:1]).split()
print(sent)

def create_sentence_windows(sentence, n=3):
    "create a sliding window over the n terms in a list of terms"

    # create a window on the first n terms by slicing the sentence into the first n terms
    window = sentence[0:n]

    # create a list to store all windows
    # add the first window that was created above
    sentence_windows = [window]

    # iterate through the rest of the terms of the sentence
    # e.g. if n=3, then create a new window with terms 2 to 4
    for term in sentence[n:]:
        # remove the first terms of the window and add the next term from the sentence
        window = window[1:] + [term]
        # add the updated window to the master list
        sentence_windows.append(window)

    return sentence_windows

# execute the function
sentence_window = create_sentence_windows(sent, n=3)
# view the first few results
sentence_window[0:5]

# execute the function for all sentences

# create a list to store all windows
sentence_window = []

for sent in matches_df['sentences']:
    # convert the sentence string into a list of terms
    sent = sent.split()

    # create the sentence windows and append to the sentence_windows list
    windows = create_sentence_windows(sent, n=3)

    # add each window to the sentence_window list
    # iterate through windows to make each item in sentence window a window, not a list of windows
    for window in windows:
        sentence_window.append(window)

# view the first five results
sentence_window[0:5]


from itertools import combinations
from collections import defaultdict

# create a defaultdict to keep track of common phrases
window_count = defaultdict(int)

for sent in sentence_window:
    # remove stop words
    sentence = [term for term in sent if term not in STOP_WORDS]

    # create a combination of terms
    # e.g. (rather, than, pay) --> (rather,than), (than,pay), (rather,pay)
    for combo in combinations(sentence, 2):
        # convert the tuple to a term
        # e.g. (rather, than) --> 'rather_than'
        phrase = '_'.join(combo)

        # increment the count for the term each time it appears to identify the most common terms
        window_count[phrase] += 1

# sort to view the most common terms
# the key (lambda x: x[1]) sorts by the count
sorted(window_count.items(), key=lambda x: x[1], reverse=True)[0:20]


# ### Phrase (collocation) Detection
#
# Phrase modeling is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that co-occur (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens $A$ and $B$ constitute a phrase is:
#

# ##### Scikit-learn API for Gensim

from gensim.sklearn_api.phrases import PhrasesTransformer

sklearn_phrases = PhrasesTransformer(min_count=3, threshold=3)
sklearn_phrases.fit(matched_sents)

#sklearn_phrases.transform(matched_sents)
print(matched_sents)


# review phrase matches
phrases = []
for terms in sklearn_phrases.transform(matched_sents):
    for term in terms:
        if term.count('_') >= 2:
            phrases.append(term)
print(set(phrases))

# create a list of stop words
from spacy.lang.en.stop_words import STOP_WORDS
common_terms = list(STOP_WORDS)


# **common_terms:** optional list of “stop words” that won’t affect frequency count of expressions containing them.
# - The common_terms parameter add a way to give special treatment to common terms (aka stop words) such that their presence between two words won’t prevent bigram detection. It allows to detect expressions like “bank of america” or “eye of the beholder”.
#

# ##### Gensim API
# A more complex API, though it is faster and has better integration with other gensim components (e.g. Phraser)

from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

phrases = Phrases(
      matched_sents
    , common_terms=common_terms
    , min_count=3
    , threshold=3
    , scoring='default'
)

phrases


# ### Phrases Params
#
# - **scoring:** specifies how potential phrases are scored for comparison to the threshold setting. scoring can be set with either a string that refers to a built-in scoring function, or with a function with the expected parameter names. Two built-in scoring functions are available by setting scoring to a string:
#
#     - ‘default’: from “Efficient Estimaton of Word Representations in Vector Space” by Mikolov, et. al.:
#
# $$\frac{count(AB) - count_{min}}{count(A) * count(B)} * N > threshold$$
#
#
#     - where N is the total vocabulary size.
#     - Thus, it is easier to exceed the threshold when the two words occur together often or when the two words are rare (i.e. small product)

# In[151]:


bigram = Phraser(phrases)

bigram


# The phrases object still contains all the source text in memory. A gensim Phraser will remove this extra data to become smaller and somewhat faster than using the full Phrases model. To determine what data to remove, the Phraser ues the  results of the source model’s min_count, threshold, and scoring settings. (You can tamper with those & create a new Phraser to try other values.)

def print_phrases(phraser, text_stream, num_underscores=2):
    """ identify phrases from a text stream by searching for terms that
        are separated by underscores and include at least num_underscores
    """

    phrases = []
    for terms in phraser[text_stream]:
        for term in terms:
            if term.count('_') >= num_underscores:
                phrases.append(term)
    print(set(phrases))


print_phrases(bigram, matched_sents)


# ### Tri-gram phrase model
#
# We can place the text from the first phrase model into another Phrases object to create n-term phrase models. We can repear this process multiple times.

phrases = Phrases(bigram[matched_sents], common_terms=common_terms, min_count=5, threshold=5)
trigram = Phraser(phrases)

print_phrases(trigram, bigram[matched_sents], num_underscores=3)

for doc_num in [5]:
    print('DOC NUMBER: {}\n'.format(doc_num))
    print('ORIGINAL SENTENT: {}\n'.format(' '.join(matched_sents[doc_num])))
    print('BIGRAM: {}\n'.format(' '.join(bigram[matched_sents[doc_num]])))
    print('TRIGRAM: {}'.format(' '.join(trigram[bigram[matched_sents[doc_num]]])))

# #### Export Cleaned Text

# write the cleaned text to a new file for later use
with open(AIRLINE_CLEANED_TEXT_PATH, 'w') as f:
    for line in bigram[matched_sents]:
        line = ' '.join(line) + '\n'
        line = line.encode('ascii', errors='ignore').decode('ascii')
        f.write(line)

# ### Advanced - clean text using SpaCy and gensim

def clean_text(doc):
    print(doc, '\n')

    ents = nlp(doc.text).ents

    # Add named entities, but only if they are a compound of more than word.
    IGNORE_ENTS = ('QUANTITY','ORDINAL','CARDINAL','DATE'
                   ,'PERCENT','MONEY','TIME')
    ents = [ent for ent in ents if
             (ent.label_ not in IGNORE_ENTS) and (len(ent) > 2)]

    # add underscores to combine words in entities
    ents = [str(ent).strip().replace(' ','_') for ent in ents]

    # clean text for phrase model
    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc_ = [token.lemma_ for token in doc if token.is_alpha]
    phrase_text = [str(term) for term in doc_]
    sent = bigram[phrase_text]
    phrases = []
    for term in sent:
        if '_' in term:
            phrases.append(term)

    # remove stops words -
    # separate step as they are needed for the phrase model
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # add phrases and entities
    doc.extend([entity for entity in ents])
    clean_text = [str(term) for term in doc] + phrases

    return clean_text

# combined terms after phrase model
after_phrase = []
for sent in doc.sents:
    text = clean_text(sent)
    for term in text:
        if '_' in term:
            after_phrase.append(term)

print(set(after_phrase))
