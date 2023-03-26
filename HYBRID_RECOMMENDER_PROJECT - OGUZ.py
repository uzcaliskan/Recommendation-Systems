
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################


# Adım 1: Movie ve Rating veri setlerini okutunuz.
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie.head()

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating.head()


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
movie.merge(rating, "left", on="movieId").head()
movie.merge(rating, "inner", on="movieId").head()
df = movie.merge(rating, "left", on="movieId")
# movie.merge(movie, "left", on="movieId").head()
# movie.merge(movie, how="inner", on="movieId").head()
df.head()
# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
oy_sayisi = pd.DataFrame(df["title"].value_counts())
rating_counts = df.groupby("title").agg({"rating":"count"}).sort_values("rating", ascending=False)

# groupby ile value counts arasındaki tek fark value_counts ile değerler yukarıdan aşağı sıralı geliyor!
oy_sayisi2 = pd.DataFrame(df.groupby("title")["rating"].count()).sort_values("rating", ascending=False)

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = oy_sayisi[oy_sayisi["title"] < 1000].index # film isimlerini aldık
common_movies = df[~(df["title"].isin(rare_movies))] #ilgili filmleri ana df den çıkardık
# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table("rating", ["userId"], ["title"], aggfunc="mean")
user_movie_df2 = common_movies.groupby(["userId", "title"])["rating"].mean().unstack()
# not: unstack ile en iyi sonucu pandas serisi veriyor. o yüzden agg kullanmadık!

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
    rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
    df = rating.merge(movie, "left", on="movieId")
    oy_sayisi = pd.DataFrame(df["title"].value_counts())
    rare_movies = oy_sayisi[oy_sayisi["title"] < 1000].index  # film isimlerini aldık
    common_movies = df[~(df["title"].isin(rare_movies))]  # ilgili filmleri ana df den çıkardık
    user_movie_df = common_movies.pivot_table("rating", ["userId"], ["title"], aggfunc="mean")
    return user_movie_df

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1).iloc[0])
# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
user_movie_df[user_movie_df.index == random_user].iloc[:, 0:5]
random_user_df = user_movie_df[user_movie_df.index == random_user]
# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
# movies_watched = random_user_df.columns[random_user_df.notna().any()]

# random_user_df.notna().any() kodu ile değişkenlerin değerlerinin herhangi biri boş değil(yani dolu mu) diye soruluyor
# tek satır df olduğundna problem oluşturmuyor

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched] # tüm kulanıclar bazında random_user in izlediği filmler

movies_watched_df.iloc[0:5, 0:5]
movies_watched_df.T.iloc[0:5, 0:5]
# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
user_movie_count = movies_watched_df.T.notnull().sum() # burada kullanıcıları sütunlara alarak her bir kullanıcı için toplam kaç tane
                                                        # film izlediklerini hesaplıyoruz
# user_movie_count: herbir kullancıını izlediği film saysı

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

# izlenne film sayısı:
len(movies_watched)
perc = len(movies_watched) * 60 / 100
user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc].sort_values("movie_count", ascending=False)

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

movies_watched_df.iloc[0:5, 0:5]

movies_watched_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies["userId"])]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
# corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = movies_watched_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()
corr_df.columns = ["user_id_1", "user_id_2", "corr"]
#corr_df[corr_df["user_id_1"] == random_user]
corr_df[corr_df["user_id_1"] == random_user].sort_values("corr", ascending=False)


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"]> 0.65)].sort_values("corr", ascending=False)
top_users.shape

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

rating.head()
top_users = top_users[["user_id_2", "corr"]]
top_users.columns = ["userId", "corr"]
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users = top_users.merge(rating, on="userId", how="left")
top_users["userId"].nunique() # çıktı: 29
top_users["movieId"].nunique() # çıkıt: 5286
top_users.head()
top_users.shape # çıktı: (13807, 5)
# Not: top users ve ratings merge işleminden sonra 29 kullanıcının 5286 filme toplam 13807 oy kullandığı görülmektedir.
#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users["weighted_rating"] = top_users["corr"] * top_users["rating"]
# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recomendation_df = top_users[["movieId", "weighted_rating"]]
recomendation_df = top_users.groupby("movieId").agg({"weighted_rating":"mean"})
recomendation_df.reset_index(inplace=True)
# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
ids = recomendation_df[recomendation_df["weighted_rating"] > 3.3].sort_values("weighted_rating", ascending=False)["movieId"].values.tolist()

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
recomendation_df = recomendation_df.merge(movie, how="left", on="movieId")
recomendation_df[recomendation_df["movieId"].isin(ids)]["title"].iloc[0:5]

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")



# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
rating.head()
id = rating[(rating["userId"] == 108170) & (rating["rating"] == 5)].sort_values("timestamp", ascending=False).iloc[0,1]
id = rating[(rating["userId"] == 108170) & (rating["rating"] == 5)].sort_values("timestamp", ascending=False)["movieId"].values[0]

name_movie = movie[movie["movieId"] == id]["title"].values[0]

user.head()
user_movie_df.iloc[0:5, 0:5]
# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

sample = user_movie_df[name_movie]
# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
# name_movies = movie[movie["movieId"]==id]["title"].values[0]
user_movie_df.corrwith(sample).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
user_movie_df.corrwith(sample).sort_values(ascending=False).head(10)[1:6].index.tolist()




