import numpy as np
import re
import spacy
from smart_open import smart_open
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

text_titles = ['Principles_of_morals_Hume', 'Political_discourses_Hume', 'Dialogues_natural_religion_Hume','Concerning_human_understanding_Hume',
'Cratylus_Plato', 'Apology_criton_phaedo_Plato', 'Gorgias_Plato', 'Republic_Plato',
 'History_peter_great_Voltaire', 'Socrates_Voltaire', 'Philosophical_dictionary_Voltaire', 'Candide_Voltaire',
 'Analysis_Mind_Russell', 'Mysticism_logic_Russell', 'Problems_philosophy_Russell', 'Roads_freedom_Russell', 
 'Pure_reason_Kant', 'Practical_reason_Kant', 'Judgment_Kant', 'Perpetual_peace_Kant']

def preprocessing(file_path = str):
    text = open(f'data_txt/{file_path}.txt').read().lower() # reads the file and convert all string to lowercase.
    replace = ['_', '°', ' ', '*', '\n', '—', '|', '\t', '\u200b', '§', 'ç', '^'] # list of symbols to be replace.
    text_replace = re.sub(f'{replace}', ' ', text) # with regex it substitutes the symbols if the list replace in the text with a space.
    text_clean = re.sub(r'(\[.+?\])|(\{.+?\})', '', text_replace) # substitutes parenthesis and curly brackets.
    print('Text cleaned, now making tokens.')
    
    nlp = spacy.load('en_core_web_sm') # loads spacy module for processing english words. 
    nlp.max_length = 10000000 # augments the length of characters the nlp object can receive. 
    stopwords = spacy.lang.en.stop_words.STOP_WORDS # loads the stopwords from spacy.
    print('Spacy completely load.')
    doc = nlp(text_clean) # converts the text into a spacy doc with many linguistic features
    text_tokenized = [token.lemma_ for token in doc if not token.is_stop] # iterates over each word in doc and checks if its not a stop word 
    print('Text converted into token lemmas.\n')                          # and return de word's lexeme.
    return text_tokenized

    
def save_txt(document):
    np.savetxt(f'data_clean/document.txt', # from numpy calls a function to save txt files.
               np.array(document), 
               newline='\n', 
               encoding='utf-8', 
               fmt="%s")
    print('Text saved in folder.')

def prepare_corpus(file_name):
    '''
    This function creates a dictionary, bag of words and TF-IDF matrix
    from a text file.
    It also saves dictionary, BOW and TF-IDF objects to disk.
    Brisa's Function
    '''
    #Create gensim dictionary
    dictionary = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open(f'data_clean/{file_name}.txt', encoding='utf-8'))
    print("Created Dictionary.\nFound {} words.\n".format(len(dictionary.values())))
    
    #Filter dictionary for common words
    #dictionary.filter_extremes(no_above=0.5, no_below=300)
    #dictionary.compactify()
    #print("Filtered Dictionary.\nLeft with {} words.\n".format(len(dictionary.values())))
    
    #Create Bag of Words
    bow = []
    for line in smart_open(f'data_clean/{file_name}.txt', encoding='utf-8'):
        tokenized_list = simple_preprocess(line, deacc=True)
        bow.append(dictionary.doc2bow(tokenized_list, allow_update=True))
    print("Created Bag of Words.\n".format(len(bow)))
    
    #Create TF-IDF Matrix
    tfidf = TfidfModel(bow, smartirs='ntc')
    tfidf_corpus = tfidf[bow]
    print("Created TF-IDF matrix.\n".format(len(tfidf_corpus)))
     
    #Save files to disk
    dictionary.save(f'dictionary_corpus/{file_name}_dictionary.dict')
    print('Saved dictionary object to disk.')
    corpora.MmCorpus.serialize(f'dictionary_corpus/{file_name}_bow_corpus.mm', bow)
    print('Saved bag of words corpus object to disk.')
    corpora.MmCorpus.serialize(f'dictionary_corpus/{file_name}_tfidf_corpus.mm', tfidf_corpus)
    print('Saved TF-IDF corpus object to disk.')
    
    return print('Processed Text')

clean_texts = [preprocessing(title) for title in text_titles]
document = [word for text in clean_texts for word in text]
save_txt(document)
prepare_corpus('document')
