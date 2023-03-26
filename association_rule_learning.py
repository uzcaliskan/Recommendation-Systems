##################
## Birliktelik Kuralı Öğrenimi - Association Rule Learning
##################
# dataset : # https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 800)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules
# pip install mlxtend
# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
#     ARL veri yapısı kullanacak olduğumz fonksiyonların beklediği veri yapısıdır
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################
df_ = pd.read_excel("Tavsiye Sistemleri/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df.describe().T

df = df_.copy()
df.head()
df.info()
########### veri okunmasında problem olması durumundan aşağıdaki kodlar
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")
df.isnull().sum()
df.describe().T
df.shape

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe
df = retail_data_prep(df)
df.describe().T
import seaborn as sns, matplotlib.pyplot as plt
# sns.boxplot(df,x=df["Quantity"])
# sns.histplot(df,x="Quantity")
# df["Quantity"].hist()
# plt.show(block=True)
# aykırı değerleri, bir değişkendeki genel dağılımın dışında olan değerlerdir

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + interquantile_range * 1.5 # aykırı değeri baskılayacağımız üst limit
    low_limit = quartile1 - interquantile_range * 1.5 # aykırı değeri baskılayacağımız alt limit
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# retail_data_prep i tekrar hazırlıyoruz
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe
df = retail_data_prep(df)
df.describe().T
############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
#      Satırlarda Invoice Sutunlarda Product olmasını istiyoruz
#      Bir faturada belirli bir ürünün olup olmamasını 0 ve 1 ile ifade ediyoruz
############################################
df.head()
####### oluşmasını istediğimiz matris( invoice lar sepet olarak değerlendirilecek)
# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# Bu matriste satılarda tüm invoice lar sütunlarda ise tüm olası ürünler yer olacaktır
# amacımız invoice x product matrisi oluşturmak

####################### ders tekrarı kodları
fr_inv_por_df = df_fr.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# fr_inv_por_df[fr_inv_por_df > 0] = 1 # güzel kod
fr_inv_por_df.isnull().sum().sum()
frequent_itemsets = apriori(fr_inv_por_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)
def check_id(dataframe, stock_code):
    dataframe[dataframe["StockCode"]==stock_code]["Description"].values[0].tolist()

df_fr[df_fr["StockCode"]==22728][["Description"]].values[0].tolist()  # burada "Description" kısmında dataframe dönmeis için çift parantez kullanıldı


rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)

rules.sort_values("support", ascending=False)

######################################### apriori ders tekrarı kodları burda bitiyor

df_fr = df[df["Country"] == "France"]
df_fr.head()
df_fr.groupby(["Invoice", "Description"])["Quantity"].sum().head(20) # pandas serisi
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head(20) # pandas dataframe
df_fr.pivot_table("Quantity",["Invoice", "Description"], aggfunc="sum").head(20)
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack() # unstack metodu group by işlemii pivot hale getiriyor!
df_fr.pivot_table("Quantity","Invoice","Description", aggfunc="sum").head(20)
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0) # boşluklara 0 yazdık.
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0)

df_fr.groupby(["Invoice", "Description"]). \
    agg({"Quantity": "sum"}). \
    unstack().fillna(0).applymap(lambda x: 1 if x >0 else 0)
# applymap() metodu ile ilgili fonksiyonu direkt olarak dataframe deki elementlere(gözlem birimlerine) uyguluyoruz.
# apply works on a row / column basis of a DataFrame
# applymap works element-wise on a DataFrame
# map works element-wise on a Series

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)  ###benim fonksiyon, pandas dataframe i


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)   # vahidonun fonksitonu - pandas serisi

type(df_fr.groupby(["Invoice", "Description"])["Quantity"].sum())
type(df_fr.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}))

# ÇOK ÖNEMLİ NOT::::: create_invoice_product_df fonksiyonunda pandas dataframe yerine pandas serisi olması apriori
# metodunda çıktının düzgün ve doğru görünmesini sağlıyor

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


#### stok kodu olan ürünün ismini liste olarak print etmek için chech_id tanımlıyoruz
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code ][["Description"]].values[0].tolist()
    print(product_name)
####### ÇOK ÖNEMLİ NOT::
# dataframe[dataframe["StockCode"] == stock_code ]["Description"].values[0].tolist # buradaki tolist çalışmasının  sebebi
# numpy array inden listeye çevrilmesidir. ["Description"]] ile pandas dataframe e çevirdik
# df[df["StockCode"] == "85123A"]["Description"][0] # sonucu str - tolist ile listeye çevrilmiyor. list parantez içine alındığında
# her harfi liste olarak gösteriyor.

df.head()
check_id(df_fr, 22728)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################
  # Esas Amacımız Burası

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True) # appriori ye pandas serisi yollamak gerekiyor
frequent_itemsets.sort_values("support", ascending=False)
frequent_itemsets.head()
#not: min_support = 0.01 yapılarak 0.01 in altındaki değerlerin gösterilmesini önlemiş olduk.
# appriori ile her bir ürünün kendine göre olasılığını hesapladık
# appriori kullanmak için ilgili fatura-ürün matirisndein çindeki değerlerin 0-1 ya da true-false olması lazım

# birliktelik kurallarını çıkarmak için;
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) # hesaplamada kullanılacak metrik

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]

# ['antecedents' = "ilk ürün",
# 'consequents' = "sonraki ürün",
# 'antecedent support' = "ilk ürünün support u",
# 'consequent support' = "sonraki ürünün support(tek başına gözlenme olasılığı) u",
# 'support' = "iki ürünün birlikte görülme olasılığı"
# 'confidence' = "bir ürün alındığında diğernin alınma olasılığı"
# "lift" = "bir ürün alındığında diğerinin alınma olasılığı lift kat artar"
# "leverage" = "kaldıraç etkisi -support u yüksek olan değere öncelik verir, yanlıdır", lift daha önemli
# 'conviction' = "y olmadan x ürürnünün beklene frekansıdır"]


############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + interquantile_range * 1.5 # aykırı değeri baskılayacağımız üst limit
    low_limit = quartile1 - interquantile_range * 1.5 # aykırı değeri baskılayacağımız alt limit
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# retail_data_prep i tekrar hazırlıyoruz
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe
retail_data_prep(df)


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)   # vahidonun fonksitonu - pandas serisi

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code ][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df.head()
rules = create_rules(df)
rules.shape

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek: bir kullanıcı var ve bu kullanıcı sepete ürün ekledi
# Kullanıcı örnek ürün id: 22492
# NOT: NORMALDE ÜRÜN ÖNERİLERİ YAPARKEN SEPETE HANGİ ÜRÜN EKLENDİĞİND HANGİ ÜRÜNÜN ÖNERİLECEĞİ BİR SQL TABLOSUNDA SAKLI TUTULUR
# BURADA YAPACAĞIMIZ TABLOLARI BİR SQL TABLOSUNA BASIP OARADA SAKLAMAK GEREKMEKTEDİR. YANİ BİR KULLANICI GELDĞİNDEN BU İŞLEMLER HEMEN YAPILMAZ
#HANGİ ÜRÜRNNE HANGİ ÜRÜN TAVSİYE EDİLECEK ÖNCESİNDEN BELLİ OLUP SQL TARAFINDAN VERİ KULLANICIYA AKTARILIR !
product_id = 22492
check_id(df, 22492)
sorted_rules = rules.sort_values("lift", ascending=False)
# NOT: BURDA LIFT E GÖRE SIRALAMA VAHİT İN YORUMUNA GÖRE YAPILMIŞTIR

recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

# NOT: RECOOMADATION_LIST OLUŞTURURUSKEN DF İÇİNDEKİ FROZENSET YAPISINDA GEZEBİLMEK İÇİN İLGİLİ DEĞİŞKENLERİ LİSTEYE ÇEVİRDİK
# Consequent değişkeninde 3-3-4 lü değerler olduğundan kolaylık olasmın açısından 0. index teki değeri getirdik

recommendation_list[0:3] #lift e göre 22492(procut_id) numaralı ürün için önerilmesi gereken ilk 3 ürün.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]
# arl : association rule learning
arl_recommender(rules, 22492, 2)
