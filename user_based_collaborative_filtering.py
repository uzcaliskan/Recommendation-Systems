############################################
# User-Based Collaborative Filtering
#############################################
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False) # çıktıları tek satırda görmek için

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
    rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    rating_counts = pd.DataFrame(df["title"].value_counts())
    rare_rated_movies = rating_counts[rating_counts["title"] <= 1000].index  # rating_counts["title"]; title:oy sayısı, film isimleri
    common_rated_movies = df[~df["title"].isin(rare_rated_movies)]
    user_movie_df = common_rated_movies.pivot_table(values="rating", index=["userId"], columns=["title"])
    return user_movie_df


user_movie_df = create_user_movie_df()
user_movie_df.head()
user_movie_df.iloc[0:5, 0:5]
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values) #kripto sinan
# sample(1, random_state=45): video da Vahido ile aynı sonucu almak için random_state=45 yapıldı

#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user_df = user_movie_df[user_movie_df.index == random_user] # kripto sinana göre veri seti indirgenmiş oldu
random_user_df.notna() # dolu mu diye soruldu
random_user_df.notna().any() #bununla pandas serisi geldi. çünkü sutunlara dolu mu diye sorumuş olduk
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() #bu kod güzel çok güzel
#burada filmlerin sütunda olması ve tek satır gözlem olması nedeniyle any() ile filmler bir pandas serisinde indexlere alınmış oldu
### Sinan'ın izlediği filmlerin Oğuzca belirlenmesi
yeni = random_user_df.T
yeni = yeni.reset_index()
yeni = yeni[yeni[28941]>0].index.tolist()
len(yeni)
################
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]

#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

### NOT: buradaşu karar verildi. sinan ile belirli bir sayı ve üzerindeki filmi izleyen kullanıclar seçilecek!. çünkü
# sinanın izlediği tüm filmleri başka kullanıcılar izlemiş olamayabilir!

user_movie_count = movies_watched_df.T.notnull().sum()
# kullanıcılar tanspoze ile değişkenlere atandı ve her bir değişken için boş olmayanların toplam saysıı bulundu
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
# "movie_count" : her bir kullanıcının toplam ne kadar film izlediği
# AMACIMIZ SİNAN İLE EN AZ AYNI 20 FİLMİ İZLEYEN KULLANICILARA ULAŞMAK

user_movie_count[user_movie_count["movie_count"] > 20].sort_values(by="movie_count",ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

#### SİNAN IN İZLEDİĞİ FİLMLERİN BELİRLİ BİR YÜZDSİNDEN SONRASI İÇİN AŞAĞIDAKİ TAKLALAR ATILABİLİR
# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                     random_user_df[movies_watched]])


final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
movies_watched_df.iloc[0:5, 0:5]
movies_watched_df.shape
# final_df ile 20 den fazla ortak film izleyenler ile sinanın izlediği filmler birleştirilmiş oluyor.
# final_df i random_user(sinan) ın bilgilerini alt alta eklemek için kulalndık
# movies_watched_df : sinanın izlediği filmlerin df i
#user_same_movies sinan ile ortak en az 20 den fazla film izleyenlerin id leri
# random_user_df : kullanıcı sinan olan sütunlarda sinanın izlediği filmler olan df
final_df.T.corr().iloc[0:5, 0:5]
final_df.T.corr().unstack().head(50)
# çok öenmli not: unstack() ile hem pivot yapılabilir hem de pivot halden groupby haline(iki index'li hale) geri dönülebilir


final_df.T.corr().unstack().sort_values(ascending=False)
final_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates() # ilk duplike değeri tutp diğerlerini siliyor!
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
## final_df.T.corr(): corr() fonksiyonu değişkenlerin birbirleriye olan korelasyonunu hesaplamak için kullanıldığından
# final_df in transpose'unu alıyoruz.
## final_df.T.corr().unstack(): unstack ile kullanıcılara göre pivot tablo oluşuyor.
##final_df.T.corr().unstack().sort_values().drop_duplicates(): sıralamadan sonra aynı korelasyonlu veriler siliniyor
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True) #drop=true ile sadece index resetlenmiş oldu
##reset_index(drop), drop : bool, default False Just reset the index, without inserting it as a column in the new DataFrame.
top_users = top_users.sort_values("corr", ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
rating.head()
top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
# top_users_rating df inde, korelasyon değerleri 0-1 arasında olduğundan korelasyon ile rating çarpılarak
# rating değerine bir düzeltme yapılmış oluyor

recommedation_df = top_users_rating.groupby("movieId").agg({"weighted_rating":"mean"})
recommedation_df = recommedation_df.reset_index()


movies_to_be_recommend = recommedation_df[recommedation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################


def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
    rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    rating_counts = pd.DataFrame(df["title"].value_counts())
    rare_rated_movies = rating_counts[
    rating_counts["title"] <= 1000].index  # rating_counts["title"]; title:oy sayısı, film isimleri
    common_rated_movies = df[~df["title"].isin(rare_rated_movies)]
    user_movie_df = common_rated_movies.pivot_table(values="rating", index=["userId"], columns=["title"])
    return user_movie_df

user_movie_df = create_user_movie_df()

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()  # bu kod güzel çok güzel
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum() #sinanın izledikleri filmlerden kullanıcılar kaçar adet izlemiş
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
    # sinan ile en az aynı 20 filmi izleyen kullanıcların id'leri
    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])
    # final_df ile en az 20 aynı filmi izlyenler ile sinanın olduğu dataframe'ler birleştiriliyor!
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    ## final_df.T.corr(): corr() fonksiyonu değişkenlerin birbirleriye olan korelasyonunu hesaplamak için kullanıldığından
    # final_df in transpose'unu alıyoruz.
    ## final_df.T.corr().unstack(): unstack ile kullanıcılara göre pivot tablo oluşuyor.
    ##final_df.T.corr().unstack().sort_values().drop_duplicates(): sıralamadan sonra aynı korelasyonlu veriler siliniyor
    corr_df = pd.DataFrame(corr_df, columns=["corr"]) # yukarıdaki corr_df in 2 index i 1 değişkeni var.
    corr_df.index.names = ["user_id_1", "user_id_2"]
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)
    top_users = top_users.sort_values("corr", ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
    top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
    top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]
    top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
    # top_users_rating df inde, korelasyon değerleri 0-1 arasında olduğundan korelasyon ile rating çarpılarak
    # rating değerine bir düzeltme yapılmış oluyor
    recommedation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"})
    recommedation_df = recommedation_df.reset_index()
    movies_to_be_recommend = recommedation_df[recommedation_df["weighted_rating"] > score].sort_values("weighted_rating",
                                                                                                     ascending=False)
    movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])
random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user,user_movie_df)

