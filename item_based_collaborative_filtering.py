###########################################
# Item-Based Collaborative Filtering
###########################################
# Dataset: https://grouplens.org/datasets/movielens/
# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
rating.head()
movie.head()
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape
df.isnull().sum()
######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.head()
df["title"].nunique()
df["title"].value_counts() # hangi filmin kaç adet puanlama aldığı görülmüş
df.groupby("title").agg({"rating": "count"}).sort_values("rating", ascending=False)

#Yorum olarak, zaman, maliyet gibi nedenlerden ötürü, puanlama sayısı
# 1000 den az olan filmleri eliyoruz
rating_counts = pd.DataFrame(df["title"].value_counts())
rating_counts[rating_counts["title"] <= 1000]
rare_rated_movies = rating_counts[rating_counts["title"] <= 1000].index # film isimleri
common_rated_movies = df[~df["title"].isin(rare_rated_movies)]
common_rated_movies["title"].nunique()
df.head()
df["title"].nunique()
common_rated_movies.head()
common_rated_movies.shape
user_movie_df = common_rated_movies.pivot_table(values="rating", index=["userId"], columns=["title"]) # values hariç değişkenleri liste içinde girince droplevel yapmaya gerek olmuyor!
user_movie_df2 = common_rated_movies.pivot_table(values="rating", index="userId", columns="title") # values hariç değişkenleri liste içinde girince droplevel yapmaya gerek olmuyor!
user_movie_df3 = common_rated_movies.groupby(["userId", "title"])["rating"].mean().unstack()
common_rated_movies.groupby(["userId", "title"])["rating"].mean().head(50)
user_movie_df3.iloc[0:5, 0:5]

user_movie_df.iloc[0:5, 0:5]
user_movie_df2.iloc[0:5, 0:5]

# user_movie_df = common_rated_movies.pivot_table("rating", index="userId", columns="title")
# NOT: user_movie_df OLUŞTURMAKTAKİ AMACIMIZ SATIRLARDA KULLANICILAR, SÜTUNLARDA FİLMLER OLAN BİR MATRIX HAZIRLAMAKTIR
user_movie_df.head()
user_movie_df.shape
######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################
movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
# sample(n): Return n random sample of items.
user_movie_df.columns[0:5]
# user_movie_df.columns = user_movie_df.columns.droplevel(0) #
movie_name = user_movie_df[movie_name]
movie_name.head(50)
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(25)
#Not: Burada, The matrix filmine kullanıcılar tarafından verilen oylar ile kullancıların diğer filmlere
# verdikleri oylar korele edilerek item temelli işbirlikçi filtreleme yapılmış oldu.!

# YANİ ÖNERİLERİ ZENGİNLEŞTİREREK ÖNERİLERİN ARKASINA KOCAMAN BİR TOPLULUĞUN FİKİR BİRLİĞİNİ ALDIK
#  İŞBİRLİKÇİ FİLTRELEME YAPTIK!

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


check_film("mat",user_movie_df)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
    rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    rating_counts = pd.DataFrame(df["title"].value_counts())
    rare_rated_movies = rating_counts[rating_counts["title"] <= 1000].index  # Dikkat! burada rating_counts["title"]; title:oy sayısı, film isimleri
    common_rated_movies = df[~df["title"].isin(rare_rated_movies)]
    user_movie_df = common_rated_movies.pivot_table(values="rating", index=["userId"], columns=["title"])
    # user_movie_df.columns = user_movie_df.columns.droplevel(0)
    return user_movie_df

user_movie_df = create_user_movie_df()

def item_based_recommender(movie_name,user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
item_based_recommender(movie_name,user_movie_df)

# not: corrwith
























