import pandas as pd
import numpy as np
import string
from langdetect import detect
import re
import spacy
import multiprocessing 
import itertools
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from langdetect import detect
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION
from gensim.models import LdaMulticore, TfidfModel
from gensim.corpora import Dictionary
from statistics import mean, stdev


class Preprocess:
    
    def __init__(self, dataframe):
        
        self.table = dataframe.reset_index().iloc[:,1:]
        
    @staticmethod
    def split_and_convert(text):
        if '$' in text:
            return float(text.split("$")[1])
        else:
            return float(text)
    
        
    @staticmethod
    def remove_noise(text):
        """
        Static function to clean app descriptions using re
        """
        text1 = re.sub("[\t\r\s]", " ",text)
        text1 = " " + text1
        text2 = re.sub(r"([ " + string.punctuation + "]+)[^a-zA-Z ]+", "\g<1> ", text1)
        return text2    
    
    def gaming_category(self):
        
        """aggregating all gaming categories"""
        
        self.table.loc[self.table["category"].str.contains("GAME"), "category"] = "GAMING"

    
    @staticmethod
    def language_detector(text):
        try:
            lang = detect(text)
            if lang=="en":
                return "en"
            else:
                return None
        except:
            return None

    
    def preprocessing(self):    
#         """
#         Iterates over app descriptions and using langdetect detects non-english descriptions. Collects the index
#         of non english descriptions and removes them from the self.table object. If the language is english, it applies 
#         regular expressions to eliminate unusual charachters like emojis or weird symbols.
#         """
        nlp = spacy.load("en")
    
        self.table = self.table.reset_index().iloc[:,1:]
    
        clean_text = [] # here I put the clean text app descriptions
        for i in range(len(self.table.description.tolist())):
            # putting missing value to non-english descriptions
            if self.language_detector(self.table.description.tolist()[i])=="en":
                document = self.remove_noise(self.table.description.tolist()[i])
                clean_text\
.append(" ".join([token.lemma_ for token in nlp(document) if token.pos_ in {'NOUN','ADJ','ADV','VERB'} and token.lemma_ not in {'-PRON-','https:/','_', '-', 'www', 'com', 'http:/'}]))
            
            elif self.language_detector(self.table.description.tolist()[i])==None:
                clean_text.append(np.nan)  
            
            else:
                clean_text.append(np.nan)
                                                                       
        assert len(self.table)==len(clean_text)
        self.table['clean_text'] = clean_text
        self.table.dropna(axis=0, subset=['clean_text'], inplace=True)
        return self.table
    
    def stats_preprocessing(self):
        """
        counts for both raw description and preprocessed description the number of words and the number of types - or 
        unique words.
        """
        output = {'before_tot':[],
                  'before_unique':[],
                  'after_tot':[],
                  'after_unique':[]}
        for i in range(len(self.table)):
            description_raw = self.table.description.iloc[i].split(' ')
            clean_txt = self.table.clean_text.iloc[i].split(' ')

            output['before_tot'].append(len(description_raw))
            output['before_unique'].append(len(set(description_raw)))
            output['after_tot'].append(len(clean_txt))
            output['after_unique'].append(len(set(clean_txt)))
        
        print("""Before preprocessing a description had on average {0} words with standard deviation {1}. \n
Moreover, the average of unique words was {2} and the standard deviation {3}."""\
              .format(round(mean(output['before_tot']), 2), round(stdev(output['before_tot']), 2), 
                      round(mean(output['before_unique']), 2), round(stdev(output['before_unique'])), 2))
        
        print("""\nAfter preprocessing a description has on average {0} words with standard deviation {1}. \n                
The average of unique words is now {2} and the standard deviation {3}."""\
              .format(round(mean(output['after_tot']), 2), round(stdev(output['after_tot']), 2), 
                      round(mean(output['after_unique']),2), round(stdev(output['after_unique']), 2)))

        return output
    
    @staticmethod
    def graph_statistics(dictionary):
        if not isinstance(dictionary, dict):
            raise Exception("input object is not a dictionary")
            
        sns.set(style="darkgrid") #setting the backfont style
        sns.set_context('poster',font_scale=0.8) # setting the size and the font scale
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,12))
        plt.subplots_adjust(bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.2) # adjusting the space between subplots
        plt.suptitle("Preprocessing Distributions", size = 35) # general title
        sns.distplot(dictionary['before_tot'],  hist=False, color="blue", label = "total words before", ax=ax1)
        sns.distplot(dictionary['before_unique'], hist=False,  kde_kws={'linestyle':'--'},
                     color="blue", label = "unique words before", ax=ax1)

        sns.distplot(dictionary['after_tot'],  hist=False, color="red", label = "total words after", ax=ax2)
        sns.distplot(dictionary['after_unique'], hist=False,  kde_kws={'linestyle':'--'},
                     color="red", label = "unique words after", ax=ax2)
        
        
    def words_similarity(self):
        sim={}
        corpus = [doc.split(" ") for doc in self.table.clean_text]

        w2v_model = Word2Vec(size=300, 
                             window=15, 
                             hs=0,
                             sample=0.000001,
                             negative=5, 
                             min_count=100,
                             workers=-1, 

                             iter=100)
        w2v_model.build_vocab(corpus)
        w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
        my_words = w2v_model.wv.index2word

        for combination in itertools.combinations(my_words, 2):
            if combination not in sim:
                sim[", ".join(combination)] = w2v_model.wv.similarity(combination[0], combination[1])

        return sim
    
    @staticmethod
    def most_similar_words(dictionary, k:float, type_similarity:str):
        
        """Given a dictionary -output of words_similarity- it returns the most semantically related words, higher than a 
        threshold k. The similarity type can be negative (negative semantical relationship), positive (positive semantical
        relationship) or both."""
        
        if not(isinstance(type_similarity, str) and type_similarity in ['positive', 'negative', 'both']):
            raise ValueError("Type similarity must be a string in ['positive', 'negative', 'both']")
            
        if not(isinstance(k, float) and k >= -1 and k <= 1):
            raise ValueError("K must be a float bounded by -1 and 1.")
        
        if type_similarity == "positive":
            assert k>=0, "If type_similarity is positive, k must be non-negative!"
            return {key: value for key, value in dictionary.items() if value > k}
        elif type_similarity == "negative":
            assert k<=0, "If type_similarity is negative, k must be non-positive!"
            return {key: value for key, value in dictionary.items() if value < k}
        else:
            return {key: value for key, value in dictionary.items() if abs(value) > abs(k)}
            
            
    def tfidf_all_corpus(self, min_freq:float=0.001, max_freq:float=0.75, ngram:int=1, output:str='graph'):
        
        
        """Tfidf function that computes TFIDF -obtained by word frequency and inverse document frequency- for the
        whole corpus.
        Input: 
        min_freq = float between 0 and 1, default 0.001
        max_freq = float between 0 and 1, default 0.75
        ngram = integer representing the ngrams to consider: default is 1 for unigram, 2 returns TFIDF for bigrams
        Output:
        - if output='graph' returns a barplot of the top tfidf-frequent 10 ngrams
        - if output='table' returns a pandas dataframe
        """

        if not(isinstance(min_freq, float) and min_freq < 1 and min_freq > 0):
            raise ValueError("Min_freq must be a float between 0 and 1")
        if not(isinstance(max_freq, float) and max_freq < 1 and max_freq > 0):
            raise ValueError("max_freq must be a float between 0 and 1")
        if not(isinstance(ngram, int) and ngram >= 1):
            raise ValueError("ngram must be an integer greater or equal than 1.")
        if not(isinstance(output, str) and output in ['graph','table']):
            raise ValueError("Select your output type: table or graph?")

        document=self.table.clean_text.tolist() 

        tfidf_vectorizer = TfidfVectorizer(ngram_range = (ngram, ngram), 
                                           analyzer='word', 
                                           min_df=min_freq, 
                                           max_df=max_freq, 
                                           stop_words='english', 
                                           sublinear_tf=True)

        X = tfidf_vectorizer.fit_transform(document)
        vectorizer = CountVectorizer(ngram_range=(ngram, ngram),
                                     analyzer = "word",
                                     min_df = min_freq,
                                     max_df = max_freq,
                                     stop_words = "english")
        X2 = vectorizer.fit_transform(document)
        word_counts = X2.toarray()
        word_tfidf = X.toarray()
        word_tfidf[word_tfidf < 0.2] = 0 # setting to 0 too low frequent words
        df = pd.DataFrame(data = {"word": vectorizer.get_feature_names(),
                             "tf": word_counts.sum(axis = 0),
                             "idf": tfidf_vectorizer.idf_,
                             "tfidf": word_tfidf.sum(axis = 0)})
        df.sort_values(["tfidf", "tf", "idf"], ascending = False, inplace=True)

        if output=='graph':
            # showing the top 10 ngrams
            df=df.iloc[:10,]
            sns.set_context('poster') 
            plt.subplots(figsize=(20,10))
            graph1 = sns.barplot(x=df['word'], y=df['tfidf'], palette="rocket") 
            graph1.set_xticklabels(labels = df['word'], rotation=30)
            graph1.set_ylabel("TFIDF",fontsize=40)
            graph1.set_xlabel("")
            graph1.set_title('Top ten {}-grams'.format(ngram), fontsize=50)

        else:
            return df.reset_index().iloc[:,1:]
        
        
    def tfidf_category(self, min_freq:float=0.001, max_freq:float=0.75, ngram:int=1, output:str='graph'):
        
        """Tfidf function that computes TFIDF -obtained by word frequency and inverse document frequency- for
        a specific app category.
        Input: 
        min_freq = float between 0 and 1, default 0.001
        max_freq = float between 0 and 1, default 0.75
        ngram = integer representing the ngrams to consider: default is 1 for unigram, 2 returns TFIDF for bigrams
        Output:
        - if output='graph' returns a barplot of the top tfidf-frequent 10 ngrams
        - if output='table' returns a pandas dataframe
        """
    
        if not(isinstance(min_freq, float) and min_freq < 1 and min_freq > 0):
                raise ValueError("Min_freq must be a float between 0 and 1")
        if not(isinstance(max_freq, float) and max_freq < 1 and max_freq > 0):
            raise ValueError("max_freq must be a float between 0 and 1")
        if not(isinstance(ngram, int) and ngram >= 1):
            raise ValueError("ngram must be an integer greater or equal than 1.")
        if not(isinstance(output, str) and output in ['graph','table']):
            raise ValueError("Select your output type: table or graph?")

        categories =  self.table.category.unique()
        print("Select an input among", categories)
        k = input()
        if k not in categories:
            raise ValueError("Input must be a single string among the categories")

        document = self.table[self.table.category==k].clean_text.tolist()

        tfidf_vectorizer = TfidfVectorizer(ngram_range = (ngram, ngram), 
                                               analyzer='word', 
                                               min_df=min_freq, 
                                               max_df=max_freq, 
                                               stop_words='english', 
                                               sublinear_tf=True)

        X = tfidf_vectorizer.fit_transform(document)
        vectorizer = CountVectorizer(ngram_range=(ngram, ngram),
                                     analyzer = "word",
                                     min_df = min_freq,
                                     max_df = max_freq,
                                     stop_words = "english")
        X2 = vectorizer.fit_transform(document)
        word_counts = X2.toarray()
        word_tfidf = X.toarray()
        word_tfidf[word_tfidf < 0.2] = 0 # setting to 0 too low frequent words
        df = pd.DataFrame(data = {"word": vectorizer.get_feature_names(),
                             "tf": word_counts.sum(axis = 0),
                             "idf": tfidf_vectorizer.idf_,
                             "tfidf": word_tfidf.sum(axis = 0)})
        df.sort_values(["tfidf", "tf", "idf"], ascending = False, inplace=True)

        if output=='graph':
            # showing the top 10 ngrams
            df=df.iloc[:10,]
            sns.set_context('poster') 
            plt.subplots(figsize=(20,10))
            graph1 = sns.barplot(x=df['word'], y=df['tfidf'], palette="rocket") 
            graph1.set_xticklabels(labels = df['word'], rotation=30)
            graph1.set_ylabel("TFIDF",fontsize=40)
            graph1.set_xlabel("")
            graph1.set_title('Top ten {0}-grams in {1}'.format(ngram, " ".join(k.split("_")).capitalize()), fontsize=40)

        else:
            return df.reset_index().iloc[:,1:]