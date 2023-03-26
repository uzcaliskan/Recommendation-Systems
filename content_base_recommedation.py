#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

pd.set_option("display.width", 800)
pd.set_option("display.expand_frame_repr", False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Tavsiye Sistemleri/movies_metadata.csv", low_memory=False) #DtypeWarning kapamak için
df.head()
df.shape
df["overview"].head()
tfidf = TfidfVectorizer(stop_words="english") #ingilizce dilinde yaygın değerler olarka kullanılan(nd, an, a, the vs)
                                            # ve ölçüm taşımanaan kelimeleri sil

df.isnull().sum()
df["overview"] = df["overview"].fillna(" ") # NaN değerler ölçüm problemi çıkaracağından boşluk ile dğeiştridik

tfidf_matrix = tfidf.fit_transform(df["overview"]) #fit kısmı, ilgili işlemi yapıyor, tansform kısmı eski değerlere işlem sonucunu atıyor
tfidf_matrix.shape # çıktı: (45466, 75827)
# (45466, 75827)
#satırlarda overwiew değerleri var, sütunlarda bu değerlerdeki eşsiz kelimeler var
tfidf.get_feature_names_out() # sütun kısımları
tfidf_matrix.toarray() # satır ve sütunların kesişimlerindeki tfidf skorları
#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

# tfidf_matrix ile ilgili metinlerin matsirisi oluşturduk. şimdi bu matristen ilgili kelimelerin uzaklıkları
# hesaplanarak hangi filmlerin benzer olduğunu bulmak gerekiyor

# cosine_sim = cosine_similarity(tfidf_matrix)
tfidf_matrix = tfidf_matrix.astype("float32")
cosine_simm = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

# cosine_simm ile filmlerin içeriklerini birbirleri arasıdnaki benzerlik skorunu matris oalrak elde ediyoruz
# cosine_similarityile satır ve sütunlar filmlerin içeriklerini teksil etmektedir. tfidf_matrix 2 defa
# kullanıcldığından hem satır hem de sütunlarda film içerikleri(overview) kısımları olmaktadır!
# not: cosine_similarity(tfidf_matrix,tfidf_matrix) hesaplarken bilgisayar hafızanda kullanılacak memory kısmını
# 16-40000 arası ayarladıktan sonra çalıştı
# mustafa germeç'ten
# Matrisin hafızadaki boyutunu yarıya indirme
# import numpy as np
# tfidf_matrix = tfidf_matrix.astype(np.float32)
#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################
indices = pd.Series(df.index, index=df["title"])
indices.head()
indices.index.value_counts() # burada herbir filmden 1 den fazla sayısa olduğu görülmektedir.
indices = indices[~indices.index.duplicated(keep="last")]
# indices.index.duplicated(keep="last") - bununla tekrar eden tüm gözlemler bulundu ve sonuncunsuna last etiketi vuruldu.
# indices.index.duplicated(keep="last") kodu ile elimizdeki duplike olanlar tutuldu
indices["Cinderella"]
df.head()

movie_index = indices["Sherlock Holmes"]

cosine_simm[movie_index]
similarity_scores = pd.DataFrame(cosine_simm[movie_index], columns=["score"])
# similarity_score'da movie_index teki filmlerin diğer filmlerle olan benzerliği gelmiş oluyor

similar_movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index #0.index'te filmin kendisi var!

df["title"].iloc[similar_movie_indices]
df["title"][similar_movie_indices]
#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_simm, dataframe):
    #indexleri oluşturma
    indices = pd.Series(df.index, index=df["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    #title ın index'ini yakalama
    movie_index = indices[title]
    #title'a göre benzerlik skorlarını hesaplama
    #burada bütün filmler ile benzerlik skorları hesapnıalıyor
    similarity_scores = pd.DataFrame(cosine_simm[movie_index], columns=["score"])
    #kendisi hariç ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe["title"].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_simm, df)
content_based_recommender("The Matrix", cosine_simm, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words="english")
    dataframe["overview"] = dataframe["overview"].fillna(" ")
    #burda kelimelerin TF-IDF matrisi oluşturuluyor
    tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender("The Dark Knight Rises", cosine_sim, df)





