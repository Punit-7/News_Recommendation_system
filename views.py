from django.shortcuts import render,get_object_or_404,redirect
from .models import Data,rate
import random
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk,string,re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3


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

    return render(request,'content.html',{'obj': obj})
def getRating(request):
    if request.method=='POST':
        rating=request.POST['rating']
        articleId = request.POST['articleId']
        userId = request.POST['userId']
    r=rate(rating=rating,articleId=articleId,userId=userId)
    r.save()

def recommend(request):


    def main():
        def news_articles():
            con = sqlite3.connect(r"C:\Users\punit7\PycharmProjects\Sabudh\News_Recommendation_system-main\db.sqlite3")
            cur = con.cursor()

            df = pd.read_sql_query("SELECT * from newsapp_data", con)
            df = pd.DataFrame(df, columns=['id', 'Title', 'Author', 'Content'])

            con.close()
            return df

        def ratingsdf():
            con = sqlite3.connect(r"C:\Users\punit7\PycharmProjects\Sabudh\News_Recommendation_system-main\db.sqlite3")
            cur = con.cursor()

            df = pd.read_sql_query("SELECT * from newsapp_rate", con)
            df = pd.DataFrame(df, columns=['userId', 'articleId', 'rating'])

            con.close()
            return df

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
        df['article'] = df['article'].apply(
            lambda x: [[lemmatizer.lemmatize(wrd.strip()) for wrd in sent] for sent in x])

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

        def find_similar_movies(news_id, X, k, metric='cosine', show_distance=False):

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

        # news_titles = dict(zip(df['id'], df['Title']))
        # news_id = user1['articleId'].tail(1).item()
        # print(news_id)
        #
        # similar_ids = find_similar_movies(news_id, X, k=10)
        # news_title = news_titles[news_id]
        #
        # print(f"Because you read {news_title}")
        # for i in similar_ids:
        #     print(news_titles[i])
        #
        news_titles = dict(zip(df['id'], df['Title']))

        news_id = user1['articleId'].tail(1).item()
        print(news_id)
        similar_ids = find_similar_movies(news_id, X, k=10, metric="euclidean")
        news_title = news_titles[news_id]
        print(f"Because you read {news_title}:")
        for i in similar_ids:
            print(news_titles[i])
        main()
    return redirect('/')
