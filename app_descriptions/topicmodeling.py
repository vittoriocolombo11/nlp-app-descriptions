import numpy as np
from gensim.models import LdaMulticore, TfidfModel
from gensim.corpora import Dictionary
import multiprocessing
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

class TopicModeling:
    
    def __init__(self, df):
        self.table=df
    
    def lda(self, cat_list:list, below:int=100, above:float=0.1, eta:float=0.9):
        
        assert set(cat_list).issubset(set(self.table.category.unique()))
        
        df_topic2=self.table[self.table.category.isin(cat_list)].reset_index().iloc[:,1:]
        instances = df_topic2.clean_text.apply(str.split)
        d = Dictionary(instances)
        print("Dictionary is:",d)
        d.filter_extremes(no_below=below, no_above=above)
        print("Dictionary after filtering:",d)
        ldacorpus = [d.doc2bow(text) for text in instances]
        tfidfmodel = TfidfModel(ldacorpus)
        model_corpus = tfidfmodel[ldacorpus]
        num_topics = len(df_topic2.groupby(['category']).count())
        temp = df_topic2.groupby(['category']).count()
        prior_probabilities = temp["app"] / temp["app"].sum()
        alpha = prior_probabilities.values
        print("Prior probabilities of the topics -alpha- are:", alpha)
        num_passes = 10
        chunk_size = len(model_corpus) * num_passes/200
        print("Preliminary steps to prepare the model done")
        model = LdaMulticore(num_topics=num_topics, # number of topics
                             corpus=model_corpus, # what to train on 
                             id2word=d, # mapping from IDs to words
                             workers=min(10, multiprocessing.cpu_count()-1), # choose 10 cores, or whatever computer has
                             passes=num_passes, # make this many passes over data
                             chunksize=chunk_size, # update after this many instances
                             alpha=alpha,
                             eta=eta,
                             random_state=5
                            )
        print("Model is ready")
        topic_corpus = model[model_corpus]
        topic_sep = re.compile(r"0\.[0-9]{3}\*") 
        model_topics = [(topic_no, re.sub(topic_sep, '', model_topic).split(' + ')) for topic_no, model_topic in
                        model.print_topics(num_topics=num_topics, num_words=5)]

        descriptors = []
        for i, m in model_topics:
            print(i+1, ", ".join(m[:3]))
            descriptors.append(", ".join(m[:2]).replace('"', ''))
        print(descriptors)
        scores = [[t[1] for t in topic_corpus[entry]] for entry in range(len(instances))]
        topic_distros = pd.DataFrame(data=scores, columns=descriptors)
        topic_distros['category'] = df_topic2['category'] 
        #%matplotlib inline

        print("Preparing graph")

        sns.set_context('poster') 

        fig, ax = plt.subplots(figsize=(20,10)) 

        aggregate_by_category = topic_distros.groupby(topic_distros.category).mean()

        aggregate_by_category[descriptors].plot.bar(ax=ax);

        fig.set_size_inches(30, 30)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 25})