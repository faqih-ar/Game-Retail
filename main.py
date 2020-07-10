#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score


# In[2]:


#Read Dataset Steam
steam = pd.read_csv('steam.csv')
df = pd.DataFrame(data=steam)
#Read Dataset Steam_Description
desc = pd.read_csv('steam_description_data.csv')
df2 = pd.DataFrame(data=desc)
#Gabung Dataset Steam & Description
df2new=df2.rename(columns={"steam_appid":"appid"})
df= pd.merge(df, df2new, on='appid', how='outer')
df = df.dropna()


# In[3]:


#Seleksi Dataset
df = df[df.categories.str.contains("Online Multi-Player")]
df = df.loc[(df[['positive_ratings', 'negative_ratings','average_playtime','median_playtime']] != 0).all(axis=1)]
df.reset_index(inplace = True, level = 0, col_level=1)
img = pd.read_excel('img_data.xlsx')
dfimg = pd.DataFrame(data=img)
dfimg.columns=['appid','img']
df = pd.concat([df,dfimg['img']], axis=1, join="outer")
df['name'] = df['name'].str.title()


# In[4]:


#Memilih Datasset Bertipe String
feat = []
for i in range(len(df)):
  feat.append([])
  feat[i].append(df['name'][i])
  feat[i].append(df['developer'][i])
  feat[i].append(df['categories'][i])
  feat[i].append(df['steamspy_tags'][i])
  feat[i].append(df['short_description'][i])


# In[5]:


#Gambung Dataset Bertipe String Menjadi Satu
feat = pd.DataFrame(data=feat)
feat['combined'] = feat.values.tolist()


# In[6]:


#Menghapus Simbol Pada Dataset Bertipe String
feat_clr = []
for i in range(len(feat)):
  feat_clr.append([])
  feat_clr[i].append(re.sub('[^A-Za-z0-9]+', ' ', str(feat['combined'][i])))

feat['cleared']= feat_clr
strtolist = []
for i in range(len(feat)):
  strtolist.append(' '.join([str(elem) for elem in feat['cleared'][i]]))

feat['cleared']= strtolist


# In[7]:


#Vectorizer Memecah Kata
from sklearn.feature_extraction.text import CountVectorizer
Ve = CountVectorizer()
Ve_matrix = Ve.fit_transform(feat['cleared'])


# In[8]:


#Cosine Similarity
cosine_sim = cosine_similarity(Ve_matrix)


# In[9]:


#MinMaxScaler Untuk Data Bertipe Numerik
ft = []
ft = pd.DataFrame(data=ft)
ft["Positive_rating"] = df.iloc[:,13]
ft["Negative_rating"] = df.iloc[:,14]
ft["Average_playtime"] = df.iloc[:,15]
ft["Median_playtime"] = df.iloc[:,16]
pre = MinMaxScaler().fit_transform(ft)


# In[10]:


#Fungsi Menampilkan Semua Data Berdasarkan Nama
def show_all():
   return df.sort_values(by=['name'])


# In[11]:


#Menampilkan Index
indices = pd.Series(df.index, index=df['name']).drop_duplicates()


# In[12]:


#Fungsi Rekomendasi
def recommendation(name):
  idx = indices[name]

  #Cosine Similarity Score
  sim_score = list(cosine_sim[idx])

  sc=[]
  for i in range(len(sim_score)):
    sc.append([])
    sc[i].append(sim_score[i])
  
  #Menggabungkan Cosine Similarity Score dengan Hasil MinMaxScaller
  Cos = pd.DataFrame(data=sc, columns=['Cosin_Sim'])
  feature = pd.DataFrame(data=pre, columns=['Positive_Rating','Negative_Rating','Average_Playtime','Median_Playtime'])
  feature['Cosine_Score'] = Cos

  #Model Cluster K-Means
  K = KMeans(n_clusters=2).fit(feature)
  df['hasil'] = K.predict(feature)
  
  #Mencari Jarak Setiap Data Ke Pusat Cluster
  r = euclidean_distances(feature)
  jrk = list(r[idx])
  
  jarak = pd.DataFrame(data=jrk, columns=['Jarak Data'])
  feature['jarak ed']= jarak

  df['jarak'] = feature.iloc[:,5]
  recomendasi = df.sort_values(by=['jarak'])
  return recomendasi[:7]


# In[13]:


def show_info(name):
    idx = indices[name]
    return df.iloc[idx]


# In[14]:


def search_game(name):
    name = name.lower()
    df['name'] = df['name'].str.lower()
    dfhasil = df[df.name.str.contains(name)]
    dfhasil['name'] = dfhasil['name'].str.title()
    df['name'] = df['name'].str.title()
    return dfhasil


# In[ ]:


from flask import Flask, render_template, Blueprint, request, redirect, url_for
from flask_paginate import Pagination, get_page_parameter, get_page_args

app = Flask(__name__)
sgame = df.sort_values(by=['name'])


def get_game(offset=0, per_page=10):
    return sgame[offset: offset + per_page]

@app.route('/', methods=['GET', 'POST'])
def home():
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(sgame)
    pagination_game = get_game(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    pp = len(pagination_game)
    
    if request.method == 'POST':
        search = request.form.get('search')
        return redirect(url_for('searching', search=search))
    
    return render_template("index.html", uwow = pagination_game, page=page,
                           per_page=per_page, pagination=pagination, pp=pp)

@app.route('/game/')
def halaman_game():
    vargame = request.args.get('vargame')
    search = request.args.get('search')
    info = show_info(vargame)
    desc = info['short_description'].lstrip()
    cat = info['categories'].replace(";", ", ")
    stp = info['steamspy_tags'].replace(";", ", ")
    
    
    reko = recommendation(vargame)
    return render_template("game.html", vargame=vargame, info=info, desc=desc, cat=cat,
                          stp=stp, reko=reko, search=search)

@app.route('/search/')
def searching():
    search = request.args.get('search')
    cari = search_game(search)
    totcari = len(cari)
    return render_template("search.html", search=search, cari=cari, totcari=totcari)

if __name__ == '__main__':
    app.run()


# In[ ]:




