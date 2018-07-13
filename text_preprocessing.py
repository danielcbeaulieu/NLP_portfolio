
# coding: utf-8

# # Text Preprocessing Solution
#
# ##### Author: Daniel Beaulieu | danbeaulieu@gmail.com

# 1. SpaCy
# 2. Text Tokenization, POS Tagging, Parsing, NER
# 3. Text Rule-based matching
# 4. Text Pipelines
# 5. Advanced SpaCy Examples
import os
from IPython.core.display import display, HTML
from IPython.display import Image
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../../config.ini')
DB_PATH = config['DATABASES']['PROJECT_DB_PATH']

# confirm DB_PATH is in the correct db directory, otherwise the rest of the code will not work
DB_PATH

# check for the names of the tables in the database
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(DB_PATH)
pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con=engine)

# read the oracle 10k documents
doc_df = pd.read_sql("SELECT * FROM Documents", con=engine)
doc_df.head(3)

# read the oracle 10k sections
df = pd.read_sql("SELECT * FROM Sections ", con=engine)
df.head(3)

# Create a dataframe named fees_df that stores the section_name for each section that includes the word fee in the section_text
# create fees_df
fees_df = df[df.section_text.str.contains('fee')].section_name

#Print the count of matched sections
# print the count of matches
print('Sections with the term fee: {}\n'.format(len(fees_df)))

#Print the first five matched sections (print the section_name)
# view the first five section names
for section in fees_df[0:5]:
    print(section)
    print()

# example text
text = df.section_text[2461]
text

# ### SpaCy
# Desktop with Development C++

# ##### SpaCy Installation
# #####  to install a convolutional neural network model:
# - python -m spacy download en_core_web_lg
#
# Tokenization|Segmenting text into words, punctuations marks etc.|
# Part-of-speech (POS) Tagging|Assigning word types to tokens, like verb or noun.|
# Dependency Parsing|	Assigning syntactic dependency labels, describing the relations between individual tokens, like subject or object.|
# Lemmatization|	Assigning the base forms of words. For example, the lemma of "was" is "be", and the lemma of "rats" is "rat".|
# Sentence Boundary Detection (SBD)|	Finding and segmenting individual sentences.|
# Named Entity Recognition (NER)|	Labelling named "real-world" objects, like persons, companies or locations.|
# Similarity|	Comparing words, text spans and documents and how similar they are to each other.|
# Text Classification|	Assigning categories or labels to a whole document, or parts of a document.|
# Rule-based Matching|	Finding sequences of tokens based on their texts and linguistic annotations, similar to regular expressions.|
# Training|	Updating and improving a statistical model's predictions.|
# Serialization|	Saving objects to files or byte strings.|

# confirm which conda environment you are using - make sure it is one with SpaCy installed
import sys
sys.executable

import spacy
from spacy import displacy

get_ipython().run_cell_magic('time', '', "\n# read in a simple (small) English language model\nnlp = spacy.load('en')\n\n# another approach:\n# import en_core_web_sm\n# nlp = en_core_web_sm.load()")

get_ipython().run_cell_magic('time', '', "\n# read in a (large) convolutional neural network model\n# this will only work after the CNN model is downloaded (~800MB)\n# e.g. python -m spacy download en_core_web_lg\nnlp = spacy.load('en_core_web_lg') ")

# instantiate the document text
doc = nlp(text)

# view the text from the SpaCy object
doc
# which the SpaCy document methods and attributes
print(dir(doc))


# ### NLP Pipeline
#
# Removing unnecessary steps for a given nlp can lead to substantial descreses in processing time.
 SpaCy pipeline

# ### Tokenization
#
# SpaCy first tokenizes the text, i.e. segments it into words,

# ### Part-of-speech (POS) Tagging
# Annotation | Description
# :----- |:------|
# Text |The original word text|
# Lemma |The base form of the word.|
# POS |The simple part-of-speech tag.|
# Tag |The detailed part-of-speech tag.|
# Dep |Syntactic dependency, i.e. the relation between tokens.|
# Shape |The word shape – capitalisation, punctuation, digits.|
# Is Alpha |Is the token an alpha character?|
# Is Stop |Is the token part of a stop list, i.e. the most common words of the language?|

# import a list of stop words from SpaCy
from spacy.lang.en.stop_words import STOP_WORDS

print('Example stop words: {}'.format(list(STOP_WORDS)[0:10]))


nlp.vocab['that']
print(dir(nlp.vocab['that']))

nlp.vocab['that'].is_stop

# search for word in the SpaCy vocabulary and
# change the is_stop attribute to True (default is False)

for word in STOP_WORDS:
    nlp.vocab[word].is_stop = True

# print column headers
print('{:15} | {:15} | {:8} | {:8} | {:11} | {:8} | {:8} | {:8} | '.format(
    'TEXT','LEMMA_','POS_','TAG_','DEP_','SHAPE_','IS_ALPHA','IS_STOP'))

# print various SpaCy POS attributes
for token in doc[0:20]:
    print('{:15} | {:15} | {:8} | {:8} | {:11} | {:8} | {:8} | {:8} |'.format(
          token.text, token.lemma_, token.pos_, token.tag_, token.dep_
        , token.shape_, token.is_alpha, token.is_stop))


# ### Text Dependency Parsing
#
# spaCy features a fast and accurate syntactic dependency parser,
# check is document has been parsed (dependency parsing)
doc.is_parsed

print('{:15} | {:10} | {:10} | {:10} | {:35} | {:25}'.format(
    'TEXT','DEP','HEAD TEXT','HEAD POS','CHILDREN','LEFTS'))

for token in doc[0:20]:
    print('{:15} | {:10} | {:10} | {:10} | {:35} | {:25}'.format(
        token.text, token.dep_, token.head.text, token.head.pos_,
        str([child for child in token.children]),str([t.text for t in token.lefts])))


# #### NOUN CHUNCKS:
#
# | **TERM** | Definition |
# |:---|:---:|
# | **Text** | The original noun chunk text |
# | **Root text** | The original text of the word connecting the noun chunk to the rest of the parse |
# | **Root dependency** | Dependency relation connecting the root to its head |
# | **Root head text** | The text of the root token's head |

print('{:15} | {:10} | {:10} | {:40}'.format('ROOT_TEXT','ROOT','DEPENDENCY','TEXT'))

for chunk in list(doc.noun_chunks)[0:20]:
    print('{:15} | {:10} | {:10} | {:40}'.format(
        chunk.root.text, chunk.root.dep_, chunk.root.head.text, chunk.text))

# dependency visualization
# after you run this code, open another browser and go to http://localhost:5000
# when you are done (before you run the next cell in the notebook) stop this cell
displacy.serve(docs=doc, style='dep')

# Another option: show visualization in Jupyter Notebook
#displacy.render(docs=doc, style='dep', jupyter=True)

# ### Named Entity Recognition (NER)
# A named entity is a "real-world object" that's assigned a name – for example, a person, a country, a product, or a book title. spaCy can recognise various types of named entities in a document, by asking the model for a prediction.

print('{:10} | {:15}'.format('LABEL','ENTITY'))

for ent in doc.ents[0:20]:
    print('{:10} | {:50}'.format(ent.label_, ent.text))

# ent methods and attributes
print(dir(ent))

# entity visualization
# after you run this code, open another browser and go to http://localhost:5000
# when you are done (before you run the next cell in the notebook) stop this cell

displacy.serve(doc, style='ent')


# print all the distinct entities tagged as a law
print(set(ent.text for ent in doc.ents if 'LAW' in ent.label_))


# print all the distinct entities tagged as an organization
print(set(ent.text for ent in doc.ents if 'ORG' in ent.label_))

# print all the distinct entities tagged as a geopolitical entity
print(set(ent.text for ent in doc.ents if 'GPE' in ent.label_))

# ##### Collections - DefaultDict
# Usually, a Python dictionary throws a KeyError if you try to get an item with a key that is not currently in the dictionary.
# The defaultdict in contrast will simply create any items that you try to access
from collections import defaultdict

sentence = ['The','airline','baggage','fees','and','food','fees','are','outrageous']

d = defaultdict(int)  # define the type of data the dict stores
for word in sentence:
    d[word] += 1  # can add to unassigned keys

print(d)


# ##### Collections - Counter
#
# A Counter is a dict subclass for counting hashable objects.
from collections import Counter

# count the number of times each GPE appears
print(Counter(ent.text for ent in doc.ents if 'GPE' in ent.label_))


# ##### Iterrtools - combinations
#
# "The itertools module standardizes a core set of fast, memory efficient tools that are useful by themselves or in combination. Together, they form an “iterator algebra” making it possible to construct specialized tools succinctly and efficiently in pure Python.
#
# **Combinations**
# - Return r length subsequences of elements from the input iterable.
# - Combinations are emitted in lexicographic sort order. So, if the input iterable is sorted, the combination tuples will be produced in sorted order.
# - Elements are treated as unique based on their position, not on their value. So if the input elements are unique, there will be no repeat values in each combination.
from itertools import combinations

airlines = ['Southwest','American','Delta','United']
for combo in combinations(airlines, 2):
    print(combo)

# ##### Sorted
#
# sorted(iterable, key=None, reverse=False)
#
# - Return a new sorted list from the items in iterable.
# - Has two optional arguments which must be specified as keyword arguments.
# - key specifies a function of one argument that is used to extract a comparison key from each list element: key=str.lower. The default value is None (compare the elements directly).
# - reverse is a boolean value. If set to True, then the list elements are sorted as if each comparison were reversed.
#
# SOURCE: https://docs.python.org/3/library/functions.html#sorted

airlines =[('airlines2',3),('airlines3',2),('airlines1',1)]
sorted(airlines)
sorted(airlines, key=lambda x:x[1])
sorted(airlines, key=lambda x:x[1], reverse=True)
# sort based on the last character of the first term
sorted(airlines, key=lambda x:x[0][-1])

# dict to store all combinations of airlines that appear together
entity_relations = defaultdict(int)

# list to sort and count how often each entity appears
counter_entities = []

for sent in doc.sents:
    # extract entities for each sentence
    sent = nlp(sent.text)

    # store all entities tagged as an organization
    entities = [ent.text for ent in sent.ents if 'ORG' in ent.label_]

    # add the entities from the current sentence to counter_entities
    counter_entities += entities

    # create combinations and increment the count in entity_relations each time combo appears
    for combo in combinations(set(entities), 2):
        entity_relations[combo] += 1

print(Counter(counter_entities))

# view the entity pairs in descending order
sorted(entity_relations.items(), key=lambda x: x[1], reverse=True)

texts = df[df.section_text.str.contains('Southwest Airlines')].section_text
len(texts)

get_ipython().run_cell_magic('time', '', "\nall_ents = []\nentity_relations = defaultdict(int)\n\nfor doc in nlp.pipe(texts, batch_size=100, disable=['tagger','ner']):\n    # split the document into sentences\n    sentences = [sentence.text for sentence in doc.sents]\n    for sent in nlp.pipe(sentences, batch_size=10, disable=['parser','tagger']):\n        # store all entities tagged as an organization\n        entities = list(set(ent.text for ent in sent.ents if 'ORG' in ent.label_))\n        # skip sentence that do not have at least 2 entities to connect \n        if len(entities) < 2:\n            continue\n        # store all entities to use later to filter relevant entities\n        for e in entities:\n            all_ents.append(e)\n        # create mapping with all combonitions of entity pairs in the sentence\n        for combo in combinations(entities, 2):\n            entity_relations[combo] += 1\n    \nset(all_ents)")

set(all_ents)

airlines_map = {
'  Southwest':'Southwest',
'  Southwest (':'Southwest',
'  United   Airlines/':'United',
'ATA  Airlines':'ATA Airlines',
'ATA Airlines':'ATA Airlines',
'AirTran':'AirTran',
'AirTran Airways':'AirTran',
'AirTran Airways, Inc.':'AirTran',
'America West Airlines':'America West Airlines',
'American Airlines':'American',
'American Airlines, Inc.':'American',
'American Eagle Airlines':'American Eagle Airlines',
'Delta':'Delta',
'Delta Air Lines':'Delta',
'Global Airlines':'Global Airlines',
'JetBlue':'JetBlue',
'Northwest Airlines':'Northwest',
'Northwest Airlines Corporation':'Northwest',
'Northwest Airlines/ Continental Airlines':'Northwest',
'People Southwest Airlines':'People Southwest Airlines',
'People of Southwest Airlines':'People Southwest Airlines',
'Southwest':'Southwest',
'Southwest  Airlines':'Southwest',
'Southwest  Airlines  Co.':'Southwest',
'Southwest Airlines':'Southwest',
'Southwest Airlines Air Travel':'Southwest',
'Southwest Airlines Co.':'Southwest',
'Southwest Airlines, Co.':'Southwest',
'US Airways':'US Airways',
'US Airways Group':'US Airways',
'US Airways Group, Inc.':'US Airways',
'USAirways':'US Airways',
'United':'United',
'United Airlines':'United',
'the Southwest Airlines  ':'Southwest',
'the Southwest Airlines Co.':'Southwest'
}

airline_pairs = []
for items, count in entity_relations.items():
    if (items[0] in airlines_map) and (items[1] in airlines_map):
        airline1 = airlines_map[items[0]]
        airline2 = airlines_map[items[1]]
        airline_pairs.append((airline1, airline2, count))

airline_pairs

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

for airline1, airlines2, count in airline_pairs:
    G.add_edge(airline1, airline2, weight=count)

# positions for all nodes
pos = nx.spring_layout(G, k=5)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=10)

# edges
for (u, v, d) in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], width=d['weight'], alpha=.2)

# labels
nx.draw_networkx_labels(G, pos, font_size=10,  font_family='sans-serif')

plt.axis('off')
plt.show()


# ### Identify Relevant Text (Rule-based Matching)
# Finding sequences of tokens based on their texts and linguistic annotations, similar to regular expressions. We will use this to filter and extract relevant text.
# The Matcher identifies text from rules we specify
from spacy.matcher import Matcher

# create a function to specify what to do with the matching text
def collect_sents(matcher, doc, i, matches):
    """  collect and transform matching text

    :param matcher: Matcher object
    :param doc: is the full document to search for text patterns
    :param i: is the index of the text matches
    :param matches: matches found in the text
    """

    match_id, start, end = matches[i]  # indices of matched term
    span = doc[start:end]              # extract matched term

    print('span: {} | start_ind:{:5} | end_ind:{:5} | id:{}'.format(
        span, start, end, match_id))

# set a pattern of text to collect
# find all mentions of the word fees
pattern = [{'LOWER':'fees'}] # LOWER coverts words to lowercase before matching

# instantiate matcher
matcher = Matcher(nlp.vocab)

# add pattern to the matcher (one matcher can look for many unique patterns)
# provice a pattern name, function to apply to matches, pattern to identify
matcher.add('fee', collect_sents, pattern)

# pass the doc to the matcher to run the collect_sents function
matcher(doc)
# change the function to print the sentence of the matched term (span)

def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]
    print('SPAN: {}'.format(span))

    # span.sent provides the sentence that contains the span
    print('SENT: {}'.format(span.sent))
    print()

# update the pattern to look for any noun preceeding the term 'fees'
pattern = [{'POS': 'NOUN', 'OP': '+'},{'LOWER':'fees'}]
matcher = Matcher(nlp.vocab)  # reinstantiate the matcher to remove previous patterns
matcher.add('fee', collect_sents, pattern)
matcher(doc)

# change the function to collect sentences
def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]

    # update matched data collections
    matched_sents.append(span.sent)


matched_sents = []  # container for sentences
pattern = [{'POS': 'NOUN', 'OP': '+'},{'LOWER':'fees'}]
matcher = Matcher(nlp.vocab)
matcher.add('fee', collect_sents, pattern)
matcher(doc)

# review matches
set(matched_sents)

# change the function to count matches using defaultdict
def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]

    # update matched data collections
    ent_count[span.text] += 1  # defaultdict keys must use span.text not span!


ent_count = defaultdict(int)
pattern = [{'LOWER':'fees'}]
matcher = Matcher(nlp.vocab)
matcher.add('fees', collect_sents, pattern)
matcher(doc)

ent_count

# update the pattern to look for a noun describing the fee
ent_count = defaultdict(int)
pattern = [{'POS': 'NOUN', 'OP': '+'},{'LOWER':'fees'}]
matcher = Matcher(nlp.vocab)
matcher.add('fees', collect_sents, pattern)
matcher(doc)

ent_count


# # Pipeline
#
# If you have a sequence of documents to process, you should use the Language.pipe()  method. The method takes an iterator of texts, and accumulates an internal buffer, which it works on in parallel. It then yields the documents in order, one-by-one.
#
# - batch_size: number of docs to process per thread
# - n_threads: number threads to use (-1 is the default that let's SpaCy decide)
# - disable: Names of pipeline components to disable to speed up text processing.

from spacy.pipeline import Pipe
# get multiple sections with the term fees
# use SpaCy to determine what type of fees

texts = df[df['section_text'].str.contains('fees')]['section_text'].values[0:5]
get_ipython().run_cell_magic('time', '', "\nent_count = defaultdict(int) # reset defaultdict\n#doc = nlp(text[0]) #first single doc\nfor doc in nlp.pipe(texts): # ['parser','tagger','ner']\n    matcher(doc) # match on your text\n\nprint(ent_count)")

# ### SpaCy - Tips for faster processing
# You can substantially speed up the time it takes SpaCy to read a document by disabling components of the NLP that are not necessary for a given task.
# - Disable options: **parser, tagger, ner**
get_ipython().run_cell_magic('time', '', "\n# reset defaultdict\nent_count = defaultdict(int)\n\n# disable the parser and ner, as we only use POS tagging in this example\n# processing occurs ~5x faster\nfor doc in nlp.pipe(texts, batch_size=100, disable=['parser','ner']):  \n    matcher(doc) # match on your text\n\nprint(ent_count)")

get_ipython().run_cell_magic('time', '', "\nent_count = defaultdict(int) # reset defaultdict\n\n# disable the parser and ner, as we only use POS tagging in this example\n# processing occurs ~75x faster, but doesn't work as the tagger is needed for the matcher\nfor doc in nlp.pipe(texts, batch_size=100, disable=['parser','tagger','ner']):\n    matcher(doc) # match on your text\n\nprint(ent_count)")

# ### Analyze the different risk types by year
# get multiple sections with the term fees
texts = df[df['section_text'].str.contains('fees')][['filename','section_text']].values
len(texts)

# These fee types were extracted using the below code.
# and grouped similar fees

fee_types = {
    'Landing fees':'landing'
  , 'agriculture inspection fees':'agriculture'
  , 'attorneys fees':'attorneys'
  , 'attorneys’ fees':'attorneys'
  , 'bag fees':'bag'
  , 'baggage fees':'bag'
  , 'card fees':'card'
  , 'card interchange fees':'card'
  , 'card processing fees':'card'
  , 'change fees':'change'
  , 'credit card fees':'card'
  , 'credit card interchange fees':'card'
  , 'credit card processing fees':'card'
  , 'customs fees':'customs'
  , 'enplanement fees':'enplanement'
  , 'experts’ fees':'experts'
  , 'inspection fees':'inspection'
  , 'interchange fees':'interchange'
  , 'l1nding fees':'landing'
  , 'landing fees':'landing'
  , 'passenger fees':'passenger'
  , 'printing fees':'printing'
  , 'processing fees':'processing'
  , 'refund passenger fees':'refund'
  , 'security fees':'security'
  , 'service fees':'service'
  , 'user fees':'user'}


# dict get
# returns value if key is in dict, otherwise returns a value of your choice
print(fee_types.get('user fees', 'return this if the key is not in the dict'))
print(fee_types.get('not a value', 'return this if the key is not in the dict'))

# create simple matcher function and pattern

def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]

    # replace the fee type
    fee = fee_types.get(span.text, span.text)
    ent_count[fee] += 1

pattern = [{'POS': 'NOUN', 'OP': '+'},{'LOWER':'fees'}]
matcher = Matcher(nlp.vocab)
matcher.add('risk', collect_sents, pattern)
get_ipython().run_cell_magic('time', '', "\nyears = defaultdict(dict)\nfor year, text in texts:\n    ent_count = defaultdict(int)               # reset ent_count for each year\n    doc = nlp(text, disable=['parser','ner'])  # disable unnessecary components\n    matcher(doc)                               # match on your text\n    \n    for key, val in ent_count.items():\n        years[year][key] = val")

# view the fees by year|
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.heatmap(pd.DataFrame(years).T.fillna(0))

# ##### Text Matching: Avoid Multiple Term Matches
# When using rule-based matching, SpaCy may match the same term multiple times if it is part of different n-term pairs with one term contained in another. For instance, 'integration services' in 'system integration services.'
# To avoid matching these terms multiple times, we can add to the collect_sents function to check if each term is contained in the previous term

def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]
    sent = span.sent

    # lemmatize the matched spans
    entity = span.lemma_.lower()

    # explicity add the first entity without checking if it matches other terms
    # as there is no previous span to check
    if i == 0:
        ent_count[entity] += 1
        ent_sents[entity].append(sent)
        matched_sents.append(sent)
        return

    # get the span, entity, and sentence from the previous match
    # if more than one match exist
    last_match_id, last_start, last_end = matches[i-1]
    last_span = doc[last_start : last_end]
    last_entity = last_span.text.lower()
    last_sent = last_span.sent

    # to avoid adding duplicates when one term is contained in another
    # (e.g. 'integration services' in 'system integration services')
    # make sure new spans are unique
    distinct_entity = (entity not in last_entity) or (sent != last_sent)
    not_duplicate_entity = (entity != last_entity) or (sent != last_sent)

    # update collections for unique data
    if distinct_entity and not_duplicate_entity:
        ent_count[entity] += 1
        ent_sents[entity].append(sent)
        matched_sents.append(sent)

# ##### Multiple Patterns
# SpaCy matchers can use multiple patterns. Each pattern can be added to the Matcher individually with match.add and can use their own collect_sents function. Or use *patterns to add multiple patterns to the matcher at once.
atched_sents = []
ent_sents  = defaultdict(list)
ent_count = defaultdict(int)

# multiple patterns
pattern = [[{'POS': 'NOUN', 'OP': '+'},{'LOWER': 'fee'}]
           , [{'POS': 'NOUN', 'OP': '+'},{'LOWER': 'fees'}]]
matcher = Matcher(nlp.vocab)

# *patterns to add multiple patterns with the same collect_sents function
matcher.add('all_fees', collect_sents, *pattern)

texts = df[df['section_text'].str.contains('fee')]['section_text'].values[0:5]
for doc in nlp.pipe(texts, batch_size=100, disable=['ner']):
    matches = matcher(doc)

ent_count
ent_sents


# ### Subject Verb Object (S,V,O) Extraction
from numpy import nanmin, nanmax, zeros, NaN
from itertools import takewhile
from spacy.parts_of_speech import CONJ, DET, NOUN, VERB
from spacy.tokens.span import Span as SpacySpan

NUMERIC_NE_TYPES = {'ORDINAL', 'CARDINAL', 'MONEY', 'QUANTITY', 'PERCENT', 'TIME', 'DATE'}
SUBJ_DEPS = {'agent', 'csubj', 'csubjpass', 'expl', 'nsubj', 'nsubjpass'}
OBJ_DEPS = {'attr', 'dobj', 'dative', 'oprd'}
AUX_DEPS = {'aux', 'auxpass', 'neg'}

def subject_verb_object_triples(doc):
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc`` or ``spacy.Span``)

    Yields:
        Tuple[``spacy.Span``, ``spacy.Span``, ``spacy.Span``]: the next 3-tuple
            of spans from ``doc`` representing a (subject, verb, object) triple,
            in order of appearance
    """
    # TODO: What to do about questions, where it may be VSO instead of SVO?
    # TODO: What about non-adjacent verb negations?
    # TODO: What about object (noun) negations?
    if isinstance(doc, SpacySpan):
        sents = [doc]
    else:  # textacy.Doc or spacy.Doc
        sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i: verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[get_span_for_compound_noun(subj)[0] - start_i: subj.i - start_i + 1]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i: span[1] - start_i + 1]

                    yield (subj, verb, obj)

def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [tok for tok in sent
            if tok.pos == VERB and tok.dep_ not in {'aux', 'auxpass'}]

def get_subjects_of_verb(verb):
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts
             if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs

def get_objects_of_verb(verb):
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights
            if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights
                if tok.dep_ == 'xcomp')
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs

def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights
            if right.dep_ == 'conj']

def get_span_for_verb_auxiliaries(verb):
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(1 for _ in takewhile(lambda x: x.dep_ in AUX_DEPS,
                                              reversed(list(verb.lefts))))
    max_i = verb.i + sum(1 for _ in takewhile(lambda x: x.dep_ in AUX_DEPS,
                                              verb.rights))
    return (min_i, max_i)

def get_span_for_compound_noun(noun):
    """
    Return document indexes spanning all (adjacent) tokens
    in a compound noun.
    """
    min_i = noun.i - sum(1 for _ in takewhile(lambda x: x.dep_ == 'compound',
                                              reversed(list(noun.lefts))))
    return (min_i, noun.i)

triples = [(s,v,o) for s,v,o in subject_verb_object_triples(doc)]
triples
