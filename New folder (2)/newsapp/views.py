from django.shortcuts import render,get_object_or_404,redirect
from .models import Data,rate,comments
import random
from django.contrib import auth

# Create your views here.

def home(request):
    count = Data.objects.all().count()
    slice = random.random() * (count - 10)
    obj = Data.objects.all()[slice: slice + 13]

    Context = {'object2': obj[0], 'object3': obj[1], 'object4': obj[3], 'object5': obj[4], 'object6': obj[5],
               'object7': obj[6], 'object8': obj[7], 'object9': obj[8], 'object10': obj[9], 'object11': obj[10],
               'object12': obj[11], 'object13': obj[12]}

    return render(request,'home.html',Context)


def content(request,id):

    obj = get_object_or_404(Data,pk=id)
    obj1 = comments.objects.all()
    # # obj = get_object_or_404(Data, pk=articleId)
    a = []
    # for i in range(len(obj1)):
    #     a.append(obj1[i].comment)
    # print(a)
    # return render(request, 'content.html', {'obj': obj, 'a': a})

    b = {}
    for i in range(len(obj1)):
        a.append(obj1[i].comment)
        # b.update(obj1[i].userId : obj1[i].comment)
        b.__setitem__(obj1[i].comment , obj1[i].userName)
    print(b)
    # return render(request, 'content.html', {'obj': obj})
    return render(request, 'content.html', {'obj': obj, 'b': b})

    # return render(request,'content.html',{'obj': obj})



def getRating(request):
    if request.method=='POST':
        rating=request.POST['rating']
        articleId = request.POST['articleId']
        userId = request.POST['userId']
    r=rate(rating=rating,articleId=articleId,userId=userId)
    r.save()

    obj = get_object_or_404(Data,pk=articleId)
    return render(request,'content.html',{'obj': obj})


def addComment(request):
    if request.method=='POST':
        comment=request.POST['comment']
        articleId = request.POST['articleId']
        userName = request.POST['userName']
    c = comments(comment=comment,articleId=articleId,userName=userName)
    c.save()

    obj1 = comments.objects.all()
    # obj2 = auth.objects.all()
    # print(obj2.id)
    obj = get_object_or_404(Data, pk=articleId)
    a = []
    b = {}

    for i in range(len(obj1)):
        a.append(obj1[i].comment)
        # b.update(obj1[i].userId : obj1[i].comment)
        b.__setitem__(obj1[i].comment , obj1[i].userName )
    print(b)
    # return render(request, 'content.html', {'obj': obj})
    return render(request, 'content.html', {'obj': obj, 'b': b})

def recommend(request):
    if request.method == 'POST':
        user = request.POST['user']
    global a
    a = int(user)
    from sklearn.feature_extraction.text import TfidfVectorizer

    import nltk, string, re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    import warnings
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import sqlite3

    def news_articles():

        con = sqlite3.connect(r"db.sqlite3")
        cur = con.cursor()

        df = pd.read_sql_query("SELECT * from newsapp_data", con)
        df = pd.DataFrame(df, columns=['id', 'Title', 'Author', 'Content'])

        con.close()
        return df

    def ratingsdf():
        print("User : " ,a)
        con = sqlite3.connect(r"db.sqlite3")
        cur = con.cursor()

        df = pd.read_sql_query("SELECT * from newsapp_rate", con)
        df = pd.DataFrame(df, columns=['userId', 'articleId', 'rating'])

        df1 = df.loc[df['userId'] == a]
        con.close()
        print(df1)

        return df1

    from matplotlib import style
    import seaborn as sns

    style.use('fivethirtyeight')
    sns.set(style='whitegrid', color_codes=True)

    # nltk.download('stopwords')
    # stop_words = stopwords.words('english')
    # nltk.download('punkt')
    # nltk.download('all')

    df = news_articles()
    df.head(10)

    len(df['id'])

    user = pd.DataFrame(columns=['user_Id', 'Article_Id', 'ratings'])
    id = np.random.randint(1, 6, size=4831)
    user['user_Id'] = id
    user['Article_Id'] = df['id']
    user.sort_values(by=['user_Id'], inplace=True)

    p = len(user['user_Id'])
    import random
    numLow = 1
    numHigh = 6
    x = []
    for i in range(0, p):
        m = random.sample(range(numLow, numHigh), 1)
        x.append(m)

    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)
    user['ratings'] = flat_list

    df['article'] = df['Title'].astype(str) + ' ' + df['Content'].astype(str)

    # tokenize articles to sentences
    df['article'] = df['article'].apply(lambda x: nltk.sent_tokenize(x))

    # tokenize articles sentences to words
    df['article'] = df['article'].apply(lambda x: [nltk.word_tokenize(sent) for sent in x])

    # lower case
    df['article'] = df['article'].apply(lambda x: [[wrd.lower() for wrd in sent] for sent in x])

    # White spaces removal
    df['article'] = df['article'].apply(lambda x: [[wrd.strip() for wrd in sent if wrd != ' '] for sent in x])

    # remove stop words
    stopwrds = set(stopwords.words('english'))
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if not wrd in stopwrds] for sent in x])

    # remove punctuation words
    table = str.maketrans('', '', string.punctuation)
    df['article'] = df['article'].apply(lambda x: [[wrd.translate(table) for wrd in sent] for sent in x])

    # remove not alphabetic characters
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if wrd.isalpha()] for sent in x])

    # lemmatizing article
    lemmatizer = WordNetLemmatizer()
    df['article'] = df['article'].apply(lambda x: [[lemmatizer.lemmatize(wrd.strip()) for wrd in sent] for sent in x])

    # remove single characters
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if len(wrd) > 2] for sent in x])

    df['article'] = df['article'].apply(lambda x: [' '.join(wrd) for wrd in x])
    df['article'] = df['article'].apply(lambda x: ' '.join(x))
    print(df['article'][0])

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_article = tfidf_vectorizer.fit_transform(df['article'])

    top_tf_df = pd.pivot(data=user, index='user_Id', columns='Article_Id', values='ratings')
    top_tf_df.fillna(0)

    from scipy.sparse import csr_matrix

    def create_X(df):

        N = user['user_Id'].nunique()
        M = user['Article_Id'].nunique()

        user_mapper = dict(zip(np.unique(user["user_Id"]), list(range(N))))
        news_mapper = dict(zip(np.unique(user["Article_Id"]), list(range(M))))

        user_inv_mapper = dict(zip(list(range(N)), np.unique(user["user_Id"])))
        news_inv_mapper = dict(zip(list(range(M)), np.unique(user["Article_Id"])))

        user_index = [user_mapper[i] for i in user['user_Id']]
        news_index = [news_mapper[i] for i in user['Article_Id']]

        X = csr_matrix((user["ratings"], (news_index, user_index)), shape=(M, N))

        return X, user_mapper, news_mapper, user_inv_mapper, news_inv_mapper

    X, user_mapper, news_mapper, user_inv_mapper, news_inv_mapper = create_X(user)

    from fuzzywuzzy import process

    def news_finder(title):
        all_titles = df['Title'].tolist()
        closest_match = process.extractOne(title, all_titles)
        return closest_match[0]

    news_title_mapper = dict(zip(df['Title'], df['id']))
    news_title_inv_mapper = dict(zip(df['id'], df['Title']))

    def get_news_index(title):
        fuzzy_title = news_finder(title)
        news_id = news_title_mapper[fuzzy_title]
        news_idx = news_mapper[news_id]
        return news_idx

    def get_news_title(news_idx):
        news_id = news_inv_mapper[news_idx]
        title = news_title_inv_mapper[news_id]
        return title

    from sklearn.neighbors import NearestNeighbors

    def find_similar_news(news_id, X, k, metric='cosine', show_distance=False):

        neighbour_ids = []

        news_ind = news_mapper[news_id]
        news_vec = X[news_ind]
        k += 1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(X)
        if isinstance(news_vec, (np.ndarray)):
            news_vec = news_vec.reshape(1, -1)

        neighbour = kNN.kneighbors(news_vec, return_distance=show_distance)
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(news_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    user1 = ratingsdf()

    news_titles = dict(zip(df['id'], df['Title']))
    news_id = user1['articleId'].tail(1).item()
    print(news_id)

    similar_ids = find_similar_news(news_id, X, k=12)
    news_title = news_titles[news_id]

    print(f"Because you read {news_title}")
    a = []
    ids = []
    for i in similar_ids:
        a.append(news_titles[i])
        ids.append(i)

    res = {}
    for key in ids:
        for value in a:
            res[key] = value
            a.remove(value)
            break

    context = {'news_titles':news_titles,'r':a,'ids':ids,'res':res}

    # news_titles =  dict(zip(df['id'], df['Title']))
    #
    # news_id = user1['articleId'].tail(1).item()
    # print(news_id)
    # similar_ids = find_similar_news(news_id, X, k=10, metric="euclidean")
    #
    # news_title = news_titles[news_id]
    # print(f"Because you read {news_title}:")
    # for i in similar_ids:
    #     print(news_titles[i])

    return render(request,'a.html',context)

def search(request):

    if request.method == 'POST':
        s = request.POST['search']
    # -*- coding: utf-8 -*-
    """Content_Based

    Automatically generated by Colaboratory.

    Original file is located at
        https://colab.research.google.com/drive/1de5VUGLy626cNkgjapPln6YZzSL3z9pm
    """

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import string
    import nltk, string, re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import gensim
    from gensim.models import Word2Vec

    # nltk.download('stopwords')
    # stop_words = stopwords.words('english')
    # nltk.download('punkt')
    # nltk.download('all')

    df = pd.read_csv('news_articles.csv')

    df = df.drop('URL', axis=1)
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)

    df.head(10)

    df['article'] = df['Title'].astype(str) + ' ' + df['Content'].astype(str)

    # tokenize articles to sentences
    df['article'] = df['article'].apply(lambda x: nltk.sent_tokenize(x))

    # tokenize articles sentences to words
    df['article'] = df['article'].apply(lambda x: [nltk.word_tokenize(sent) for sent in x])

    # lower case
    df['article'] = df['article'].apply(lambda x: [[wrd.lower() for wrd in sent] for sent in x])

    # White spaces removal
    df['article'] = df['article'].apply(lambda x: [[wrd.strip() for wrd in sent if wrd != ' '] for sent in x])

    # remove stop words
    stopwrds = set(stopwords.words('english'))
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if not wrd in stopwrds] for sent in x])

    # remove punctuation words
    table = str.maketrans('', '', string.punctuation)
    df['article'] = df['article'].apply(lambda x: [[wrd.translate(table) for wrd in sent] for sent in x])

    # remove not alphabetic characters
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if wrd.isalpha()] for sent in x])

    # lemmatizing article
    lemmatizer = WordNetLemmatizer()
    df['article'] = df['article'].apply(lambda x: [[lemmatizer.lemmatize(wrd.strip()) for wrd in sent] for sent in x])

    # remove single characters
    df['article'] = df['article'].apply(lambda x: [[wrd for wrd in sent if len(wrd) > 2] for sent in x])

    df['article'] = df['article'].apply(lambda x: [' '.join(wrd) for wrd in x])
    df['article'] = df['article'].apply(lambda x: ' '.join(x))

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_article = tfidf_vectorizer.fit_transform(df['article'])

    """**Cosine Similarity**"""

    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_article, tfidf_article)

    # sum_of_squared_dist = []
    # K = range(1,15)
    # for k in K:
    #   knn=KMeans(n_clusters=k,init='k-means++')
    #   knn.fit(cosine_sim)
    #   sum_of_squared_dist.append(knn.inertia_)
    #
    # plt.plot(K, sum_of_squared_dist, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()

    knn = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)
    knn.fit(cosine_sim)

    def get_recommendations(title, cosine_sim=cosine_sim):

        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:11]

        news_indices = [i[0] for i in sim_scores]

        return df['Title'].iloc[news_indices]

    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    get_recommendations('US  South Korea begin joint military drill amid nuclear threat from North Korea')

    from sklearn.decomposition import PCA
    tfidf_pca = PCA(n_components=2)
    tfidf_pca_comp = tfidf_pca.fit_transform(tfidf_article.toarray())

    """**Clusters**"""
    #
    # sum_of_squared_dist = []
    # K = range(1,15)
    # for k in K:
    #   knn=KMeans(n_clusters=k,init='k-means++')
    #   knn.fit(tfidf_pca_comp)
    #   sum_of_squared_dist.append(knn.inertia_)
    # plt.plot(K, sum_of_squared_dist, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()
    import pickle
    k_means = KMeans(n_clusters=3)
    k_means.fit(tfidf_pca_comp)
    knnPickle = open('kmeans_file', 'wb')
    pickle.dump(k_means, knnPickle)
    loaded_model = pickle.load(open('kmeans_file', 'rb'))
    pred = loaded_model.predict(tfidf_pca_comp)
    plt.figure(figsize=(15, 15))
    plt.scatter(tfidf_pca_comp[:, 0], tfidf_pca_comp[:, 1], c=pred)
    # plt.show()

    # Use the loaded pickled model to make predictions
    df['tfidf'] = tfidf_article
    df['tfidf_clusters'] = pred
    df.head()

    df.tfidf_clusters.value_counts()

    top_tf_df = pd.DataFrame(tfidf_article.todense()).groupby(df['tfidf_clusters']).mean()

    for i, r in top_tf_df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([tfidf_vectorizer.get_feature_names()[t] for t in np.argsort(r)[-20:]]))

    """Search"""

    import pandas as pd
    import numpy as np
    import os
    import re
    import operator
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from collections import defaultdict
    from nltk.corpus import wordnet as wn
    from sklearn.feature_extraction.text import TfidfVectorizer

    def wordLemmatizer(data):
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        file_clean_k = pd.DataFrame()
        for index, entry in enumerate(data):

            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
                    # The final processed set of words for each iteration will be stored in 'text_final'
                    file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                    file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                    file_clean_k = file_clean_k.replace(to_replace="\[.", value='', regex=True)
                    file_clean_k = file_clean_k.replace(to_replace="'", value='', regex=True)
                    file_clean_k = file_clean_k.replace(to_replace=" ", value='', regex=True)
                    file_clean_k = file_clean_k.replace(to_replace='\]', value='', regex=True)
        return file_clean_k

    vocab = set()
    for doc in df['article']:
        vocab.update(doc.split(" "))

    vocab = list(vocab)
    tfidf = TfidfVectorizer(vocabulary=vocab)

    t1 = tfidf.fit(df.article)
    with open('t1.pk', 'wb') as fin:
        pickle.dump(t1, fin)
    tfidf_tran = tfidf.transform(df.article)
    with open('tfidf_tran.pk', 'wb') as fin:
        pickle.dump(tfidf_tran, fin)
    loaded_tfidf_tran = pickle.load(open('tfidf_tran.pk', 'rb'))

    def gen_vector_T(tokens):
        Q = np.zeros((len(vocab)))
        x = tfidf.transform(tokens)
        for token in tokens[0].split(','):
            try:
                ind = vocab.index(token)
                print("ind", ind)
                Q[ind] = x[0, tfidf.vocabulary_[token]]

            except:
                pass
        return Q

    def cosine_sim(a, b):
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cos_sim

    def cosine_similarity_T(k, query):
        preprocessed_query = preprocessed_query = re.sub("\W+", " ", query).strip()
        tokens = word_tokenize(str(preprocessed_query))
        q_df = pd.DataFrame(columns=['q_clean'])
        q_df.loc[0, 'q_clean'] = tokens
        q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
        d_cosines = []

        query_vector = gen_vector_T(q_df['q_clean'])
        for d in loaded_tfidf_tran.A:
            d_cosines.append(cosine_sim(query_vector, d))

        out = np.array(d_cosines).argsort()[-k:][::-1]
        # print("")
        d_cosines.sort()
        a = pd.DataFrame()
        for i, index in enumerate(out):
            a.loc[i, 'index'] = str(index)
            a.loc[i, 'article'] = df['article'][index]
            a.loc[i, 'Title'] = df['Title'][index]
        for j, simScore in enumerate(d_cosines[-k:][::-1]):
            a.loc[j, 'Score'] = simScore
        return a

    df1 = cosine_similarity_T(12, s)

    a = df1['Title'].tolist()
    ids = df1['index'].tolist()
    res = {}
    for key in ids:
        for value in a:
            res[key] = value
            a.remove(value)
            break

    return render(request,'search.html',{'data':res})