#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 800)
movie = pd.read_csv("Tavsiye Sistemleri/movie.csv")
rating = pd.read_csv("Tavsiye Sistemleri/rating.csv")
df = movie.merge(rating, on="movieId", how="left")
df.head()
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

#veri çok büyük olduğundan saec sample_df oluşturduk
sample_df = df[df["movieId"].isin(movie_ids)]
sample_df.head()
sample_df.shape
user_movie_df = sample_df.pivot_table("rating", ["userId"], ["title"])
user_movie_df.head()
user_movie_df.shape
# surprise kütüphanesinin istekleri!
reader = Reader(rating_scale=(1,5)) # burada rating skalasının hangi değerler arasında olduğunu belirtmek durumundayız
# surprise kütüphanesine özgün Dataframe olluşturuluyor
data = Dataset.load_from_df(sample_df[["userId",
                                       "movieId",
                                       "rating"]], reader)
# data ile surprise kütüphanesine özgü df oluşturuldu
##############################
# Adım 2: Modelleme
##############################
# makine öğrenmesi kapsamında modeller önce bir eğitim seti üzerine kurulur,
# akabinde ise modelin görmediği bir test seti üzerinde test edilir.

trainset, testset = train_test_split(data, test_size=0.25) # verinin %25'i test seti olarak bölündü.
svd_model = SVD() # matris factorization yöntemini kullanacak olduğumuz fonksiyon
svd_model.fit(trainset) # model kuruldu. svd_model matrix factorization yöntemi ile oluşturulmuş oldu!
predictions = svd_model.test(testset)

# 1 satır çıktı: (uid=94138.0, iid=541, r_ui=3.5, est=4.427463344726063, details={'was_impossible': False}
#               (userid      , itemid,  r_ui: kullanıcını verdiği gerçek rating,
#               est= modelin tahmini)
accuracy.rmse(predictions)

#rmse: Root Mean Squared Error: hata kareleri ortalamasının karekökü
# predictions kısmında kulanıcının verdiği ile estimated değerler arasında fark görülmekte olduğundan rmse kullanıdlı
# rmse ile ortalama ne kadar yapıldığını buluyoruz
# rmse ile tüm değerler arasında farkların karesi alınıp tolandıktan sonra tüm değerlerin sayısına bölünüp karekök alınacak!
svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True)

sample_df[sample_df["userId"] == 1]
##############################
# Adım 3: Model Tuning - Model Kurma
##############################
# Modeli optimize etmek için - modelin tahmin performansını artırmak için:
# SVD() metodunda gizli faktör - latent factors sayısı 20 oalrak öne tanımlı
# aynı şekilde learning rate, iterasyon sayısı(n_epochs) düzeltme değeri(reg_all) lambda gibi tüm
# hiperparametreler öntanımlıdır!
# HİPERPARAMETRE DEĞERLERİ BELİRLİ BİR ŞEKİLDE DEĞİŞTİRERERK EN DOĞRU SONUCU ELDE ETMEK DENENİR
# HİPERPARAMETRE DEĞERLERİNE BİR DEĞER LİSTESİ GÖNDERİLEREK HANGİ HİPERPARAMETRE DEĞERLERİNDE EN İYİ
# ÖĞRENME OLMUŞ BULUNMAYA ÇALIŞILIR
# iterasyon sayısı(n_epochs), lr_all = learning rate for all parameters
param_grid = {"n_epochs": [5, 10, 20],
              "lr_all": [0.002, 0.005, 0.07]}
# modeli kurdkk
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=["rmse", "mae"],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True
                  )
# GridSearchCV ile  SVD model nesnesini kullanacak, param_grid içindeki tüm kombinasyonları deneyecek,
# measures ile sonuçları rsme ye ve de mae(gerçek değerler ile tahmin değerlerinin farklarının mutalk değeri) göre
# değerlendirecek, cv=3 ile veri setini 3 e bölüp 3 defa itere edecek sekilde 2  parçasıyla model kurup, 1  parçasıyla
# kurduğu modeli test edecek bunları yaparken de n_jobs=-1 ile işlemcileri ful performasn kullacak ve de raporlarma sunacak

# modeli fit ettik
gs.fit(data)
gs.best_score["rmse"] # rmse ye göre en iyi skor
gs.best_params["rmse"] # rmse ye göre en iyi hiperparametreler(çıktı: {'n_epochs': 10, 'lr_all': 0.002})

# ÖZET OLARAK GridSearchCV İLE YAPTIĞIMIZ ŞEY, SVD MODELİNDE ÖN TANIMLI DEĞERLERİ YERİNE EN İYİ HANGİ PARAMETRELERİN
# KULLANIMASI GEREKTİĞİNİ BULMAYA ÇALIŞMAK. BU YAPARKEN DE param_grid ADINDAKİ SÖZLÜK DEĞERİNE
# HANGİ HİPERPARAMETREYE(lr_all, n_epochs vs) HANGİ LİSTEYİ VERİRSEK, GridSearchCV İLE O LİSTEDEKİLERE AİT EN İYİ
# SONUCU BULMAYA ÇALIŞIYORUZ

##############################
# Adım 4: Final Model ve Tahmin
##############################

svd_model = SVD(**gs.best_params["rmse"])
# ** kullanarka direkt olarak keyworded yapısını yani sözlük yapısını girmiş oluyoruz

data = data.build_full_trainset() # data vei setini train ve test diye ayırmak yerine tamamını aldık. hata oranımızı öğrendik çünkü. 0.93 :)
svd_model.fit(data)
svd_model.predict(uid=1.0, iid=541, verbose=True)


























