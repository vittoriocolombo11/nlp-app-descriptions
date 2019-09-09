import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering

warnings.simplefilter("ignore")

class Clusters:
    
    def __init__(self, dataframe):
        self.table = dataframe
    
    def kmeans(self, cat_list:list, ngram_range:int=2, min_freq:float=0.003, max_freq:float=0.5, components:int=300, sample_size:int=500, plot_dim:int=3):
        
        assert set(cat_list).issubset(set(self.table.category.unique()))
        if not (isinstance(ngram_range, int) and ngram_range >= 1):
            raise ValueError("Ngram_range for TF-IDF must be greater or equal than 1.")
        if not (isinstance(min_freq, float) and min_freq > 0 and min_freq < 1):
            raise ValueError("minimum word frequency for TF-IDF must be between 0 and 1.")
        if not (isinstance(max_freq, float) and max_freq > 0 and max_freq < 1):
            raise ValueError("maximum word frequency for TF-IDF must be between 0 and 1.")
        if not(isinstance(plot_dim, int) and plot_dim in [2,3]):
            raise ValueError("Dimensions to plot can only be 2 or 3, you passed", plot_dim)
        k = len(cat_list)
        
        ngram = (1, ngram_range)
    
        df = self.table[self.table.category.isin(cat_list)].reset_index().iloc[:,1:]
        
        corpus=df['clean_text'].values.tolist()

        tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram, stop_words='english', analyzer='word', min_df=min_freq, max_df=max_freq, sublinear_tf=True, use_idf=True)

        X = tfidf_vectorizer.fit_transform(corpus)
        
        if not (isinstance(components, int) and components < X.shape[1]):
            raise ValueError("Number of components passed is", components, ", shape of tf-idf matrix is", X.shape)

        X2 = TruncatedSVD(n_components=components).fit_transform(X) 

        agg = AgglomerativeClustering(n_clusters=k)
        np.random.seed(0)
        sample = np.random.choice(len(X2), replace=False, size=2500)
        agg_sample = agg.fit_predict(X2[sample])
        centroids = np.array([X2[sample][agg_sample == c].mean(axis=0) for c in range(k)])

        km = KMeans(n_clusters=k, n_jobs=-1, init=centroids)

        km.fit(X2)
        
        if not (isinstance(sample_size, int) and sample_size < X2.shape[0]):
            raise Exception("Size of random sample is", sample_size, "while length of reducted tf-idf matrix is", X2.shape[0])
        
        plot_sample = np.random.choice(len(X2), replace=False, size=sample_size)
        self.plot_vectors(X2[plot_sample], title = "App categories", labels=df['category'][plot_sample].values, dimensions=plot_dim)
        
        
    @staticmethod    
    def plot_vectors(vectors, title, labels=None, dimensions=3):
        """
        plot the vectors in 2 or 3 dimensions. If supplied, color them according to the labels
        """
        
        # set up graph
        
        sns.set_context('poster')

        fig = plt.figure(figsize=(10,10))

        df = pd.DataFrame(data={'x':vectors[:,0], 'y': vectors[:,1]})

        if labels is not None:
            df['label'] = labels
        else:
            df['label'] = [''] * len(df)
        cm = plt.get_cmap('prism') 
        n_labels = len(df.label.unique())
        label_colors = [cm(1. * i/n_labels) for i in range(n_labels)] 
        cMap = colors.ListedColormap(label_colors)

        # plot in 3 dimensions
        if dimensions == 3:
            # add z-axis information
            df['z'] = vectors[:,2]  #take that color and add it to the dataframe (under column z)
            # define plot
            ax = fig.add_subplot(111, projection='3d')
            frame1 = plt.gca() 
            # remove axis ticks
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            frame1.axes.zaxis.set_ticklabels([])

            # plot each label as scatter plot in its own color
            for l, label in enumerate(df.label.unique()):
                df2 = df[df.label == label]
                ax.scatter(df2['x'], df2['y'], df2['z'], c=label_colors[l], cmap=cMap, edgecolor=None, label=label, alpha=0.3, s=100)

        # plot in 2 dimensions
        elif dimensions == 2:
            ax = fig.add_subplot(111)
            frame1 = plt.gca() 
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])

            for l, label in enumerate(df.label.unique()):
                df2 = df[df.label == label]
                ax.scatter(df2['x'], df2['y'], c=label_colors[l], cmap=cMap, edgecolor=None, label=label, alpha=0.3, s=100)

        else:
            raise NotImplementedError()

        plt.title(title, fontsize = 20)
        plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right', borderaxespad=0.)
        plt.show()