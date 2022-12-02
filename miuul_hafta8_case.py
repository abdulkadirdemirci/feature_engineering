#################################
# İŞ PROBLEMİ                   #
"""
Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.
"""  #
# Veri Seti Hikayesi            #
"""
Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu
ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. 
Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir
"""  #
# Değişkenler                   #
"""
DeviceProtection:  Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
PaperlessBilling:  Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
InternetService:   Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
StreamingMovies:   Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineSecurity:   Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
MonthlyCharges:   Müşteriden aylık olarak tahsil edilen tutar
MultipleLines:    Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
PaymentMethod:  Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
SeniorCitizen:  Müşterinin yaşlı olup olmadığı (1, 0)
OnlineBackup:   Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
PhoneService:   Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
TotalCharges:   Müşteriden tahsil edilen toplam tutar
TechSupport:  Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV:  Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
customerId:   Müşteri İd’si
Dependents:   Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
Contract:   Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
Partner:    Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
gender:  Cinsiyet
tenure:  Müşterinin şirkette kaldığı ay sayısı
Churn :  Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

"""  #
#################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

###########################
# Görev 1 :               |
# Keşifçi Veri Analizi    |
###########################
df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

######################################################
# Adım 1: Genel resmi inceleyiniz.
# todo: totalcharges float'a çevrilmeli
"""
* müşterilerin min %80 max %90' genç.
* std yüksek (2 yıl) olmakla birlikte ortalama müşteri yaşı 2 yıl 8 ay
"""
######################################################
df.head()
df.info()
df.isnull().sum()
df.describe([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]).T
df.nunique()

df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors="coerce")


######################################################
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
######################################################
def grab_col_names(dataframe, cat_th=5, car_th=50):
    """
    ön tanımlı değerler iş problemine göre değişebilir.
    numeric  ve categorik değişkenleri tespit eder.

    Parameters
    ----------
    dataframe: dataframe
        inceleme yapılacak veri seti
    cat_th: int, default = 5
        kategorik sayılmak için sahip olunacak max sınıf sayısı
    car_th: int, default = 50
        cadinal sayılmak için sahip olunacak min sınıf sayısı

    Returns
    -------
    list: cat_cols, num_cols

    """
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "float64"] and
                dataframe[col].nunique() > car_th]

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "object"] and
                dataframe[col].nunique() < cat_th]

    print("NUMERIC COLUMNS: ", [col for col in num_cols], end="\n\n")
    print("CATEGORICAL COLUMNS: ", [col for col in cat_cols], end="\n\n\n")

    return num_cols, cat_cols


num_cols, cat_cols = grab_col_names(df)


######################################################
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
######################################################
def cat_analyzer(dataframe, cat_cols, target, plot=False):
    """
    kategorik değişkenler hakkında bilgi verir,
    dağılımların graiklerini verir

    Parameters
    ----------
    dataframe: dataframe
    cat_cols: list
        categorik değişkenler barındıran liste

    Returns
    -------
        özet bilgiler sunar. ve graik çizdirir
    """
    temp_df = df.copy()
    temp_df[target] = pd.get_dummies(temp_df[target], drop_first=True)
    cat_cols = [col for col in cat_cols if col not in "Churn"]

    for col in cat_cols:
        print(col, ":", df[col].nunique())
        data = pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                             "RATIO": dataframe[col].value_counts() / dataframe.shape[0],
                             "RATIO_TARGET": temp_df.groupby(col)[target].mean()})

        print(data, end="\n\n\n")
        if plot:
            mylabels = data.index
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.suptitle("** GENERAL DISTRIBUTION PIE GRAPHS **")
            mycolors = ["#DEF5E5", "#BCEAD5", "#9ED5C5", "#8EC3B0"]
            ax1.pie(data['RATIO'].values, labels=mylabels, autopct='%1.2f%%',
                    colors=mycolors, textprops={'fontsize': 7})
            ax1.set_title("Ratio Distibution", fontdict={"fontsize": 8})
            ax2.pie(data['RATIO_TARGET'].values, labels=mylabels, autopct='%1.2f%%',
                    colors=mycolors, textprops={'fontsize': 7})
            ax2.set_title("Target Ratio Distibution", fontdict={"fontsize": 8})
            plt.show()


cat_analyzer(df, cat_cols, "Churn", True)


def num_analyzer(dataframe, num_cols, plot=False):
    for col in num_cols:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)

        iqr = q3 - q1

        up_limit = q3 + 1.5 * iqr
        low_limit = q1 - 1.5 * iqr

        print(col, "\nALT SINIR: %.1f \nÜST SINIR: %.1f\n\n " % (low_limit, up_limit))
        if plot:
            sns.color_palette("rocket")
            sns.boxplot(data=dataframe, x=col, y="Churn", hue="gender")
            plt.show()
            sns.displot(data=dataframe, x=col, hue="gender", multiple="stack", kind="kde")
            plt.show()
            sns.displot(data=dataframe, x=col, hue="Churn", multiple="stack", kind="kde")
            plt.show()


num_analyzer(df, num_cols, True)


def get_outliers(dataframe, num_cols, limit=10):
    for col in num_cols:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)

        iqr = q3 - q1

        up_limit = q3 + 1.5 * iqr
        low_limit = q1 - 1.5 * iqr

        length = dataframe.loc[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

        if df[(df[col] < low_limit) | (df[col] > up_limit)].any(axis=None):
            print(col, " : True")
            if length > 10:
                dataframe.loc[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].sample(10)
            else:
                dataframe.loc[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)]
        else:
            print(col, " : Flase")


get_outliers(df, num_cols)


###########################################
# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)
######################################################
def target_analyzer(dataframe, cat_cols, num_cols, target):
    cat_cols = [col for col in cat_cols if col not in "Churn"]
    temp_df = dataframe.copy()
    temp_df[target] = pd.get_dummies(dataframe[target], drop_first=True)

    for col in cat_cols:
        print(temp_df.groupby(col).agg({target: ["mean", "count"]}), end="\n\n")

    print(temp_df.groupby(target).agg({"tenure": ["mean", "std", "max", "min", "count"],
                                       "MonthlyCharges": ["mean", "std", "max", "min", "count"],
                                       "TotalCharges": ["mean", "std", "max", "min", "count"]}),
          end="\n\n")


target_analyzer(df, cat_cols, num_cols, "Churn")


######################################################
# Adım 5: Aykırı gözlem analizi yapınız.
######################################################
def thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    q1 = dataframe[col_name].quantile(q=q1)
    q3 = dataframe[col_name].quantile(q=q3)

    iqr = q3 - q1

    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr

    return low_limit, up_limit


for col in num_cols:
    low_limit, up_limit = thresholds(df, col)
    print(col, f":\nALT SINIR: %.f \nÜST SINIR: %.f\nMAX DEĞER: %.f\n\n" % (low_limit, up_limit, df[col].max()))

######################################################
# Adım 6: Eksik gözlem analizi yapınız.
######################################################
import missingno as msno

msno.bar(df)
plt.show()
msno.dendrogram(df)
plt.show()
msno.heatmap(df)
plt.show()
msno.matrix(df)
plt.show()


def is_there_missing(dataframe, col_name):
    if dataframe[col_name].isnull().any(axis=None):
        return True
    else:
        return False


missing_cols = [col for col in df.columns if is_there_missing(df, col)]

df[missing_cols].isnull().sum()


def go_to_missing(dataframe, col_name):
    return dataframe.loc[dataframe[col_name].isnull()]


go_to_missing(df, "TotalCharges")

df.loc[df["tenure"] < 1]

######################################################
# Adım 7: Korelasyon analizi yapınız.
######################################################
df.corr()
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 3, as_cmap=True)
sns.heatmap(df.corr(), annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title("numerik değişkenlerin korelasyon analizi")
plt.show()
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


###########################
# Görev 2 :               |
# Feature Engineering     |
###########################

######################################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
######################################################
# knn imputer ile doldurmak
imputer = KNNImputer(n_neighbors=3)
df["new_total_charges_knn"] = imputer.fit_transform(df[["TotalCharges"]])
df[["TotalCharges", "new_total_charges_knn"]].describe().T

######################################################
# Adım 2:  Yeni değişkenler oluşturunuz
######################################################
df.head()
for col in df.columns:
    print(col, ":\n", df[col].unique())

df["NEW_TENURE_QCUT"] = pd.qcut(df["tenure"], q=4, labels=["kotu","orta","iyi","pekiyi"])
""" label coding yapmalıyım ordinal olarak """

df.loc[(df["Contract"] == 'One year') | (df["Contract"] == 'Two year'), "NEW_CONTRACT_DURATION"] = 1
df.loc[df["Contract"] == 'Month-to-month', "NEW_CONTRACT_DURATION"] = 0
""" Şirket ile uzun vadede iş yapmayı düşünenlerin churn olma olasılıgı düşüktür diye düşündüm """

fig, ax = plt.subplots()
ax = sns.displot(df["MonthlyCharges"], kind="kde", color="red", fill=True)
plt.xticks(range(20, 120, 5))
plt.axvline(80)
plt.axvline(36)
plt.show()

df["NEW_MONTHLY_CHARGES_CUT"] = pd.cut(df["MonthlyCharges"], bins=[0, 36, 80, 120],labels=["kotu","orta","iyi"])
""" aylık ücreti 36 ve 80 olan degerlerden cut uyguladım"""
"""
##################################
# ÖZELLİK ÇIKARIMI
##################################

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)
"""



######################################################
# Adım 3:Encoding işlemlerini gerçekleştiriniz.
######################################################
dff = df
df.head()
df.shape
df.drop("customerID", inplace=True, axis=1)

binary_cols = [col for col in df.columns if df[col].nunique() == 2]
binary_cols = [ col for col in binary_cols if col not in ["SeniorCitizen","Churn",
                                                          "NEW_CONTRACT_DURATION"]]
binary_cols.append("PaymentMethod")
multi_class_cols = [col for col in df.columns if 4 >= df[col].nunique() >= 3]
"""
binary_cols içindekiler phe drop_first il encode edilecek

multiple_cols içindekiler ordinalencoder ile encode edilecek

"""
# one hot encoding
for col in binary_cols:
    df[col] = pd.get_dummies(df[col], drop_first=True)
dff = df
dff = pd.concat([dff,pd.get_dummies(dff["PaymentMethod"],drop_first=True)],axis=1)

pd.get_dummies(dff["PaymentMethod"],drop_first=True)
dff.head()

# label encoding for ordinals
general_order = ['No internet service', 'No', 'Yes' ]
multipleline_order =  ['No phone service', 'No', 'Yes']
InternetService_order = ['No', 'DSL', 'Fiber optic' ]
Contract_order = ['Month-to-month', 'One year', 'Two year']
tenure_qcut_order = ['kotu', 'iyi', 'orta', 'pekiyi']
monthly_charges_order = ["kotu","orta","iyi"]
le = LabelEncoder()
le.fit(multipleline_order)
dff["NEW_MULTIPLELINES_ORDERED_ENC"] = le.transform(dff["MultipleLines"])
le.fit(general_order)
dff["NEW_OnlineSecurity_ORDERED_ENC"] = le.transform(dff["OnlineSecurity"])
dff["NEW_OnlineBackup_ORDERED_ENC"] = le.transform(dff["OnlineBackup"])
dff["NEW_DeviceProtection_ORDERED_ENC"] = le.transform(dff["DeviceProtection"])
dff["NEW_TechSupport_ORDERED_ENC"] = le.transform(dff["TechSupport"])
dff["NEW_StreamingTV_ORDERED_ENC"] = le.transform(dff["StreamingTV"])
dff["NEW_StreamingMovies_ORDERED_ENC"] = le.transform(dff["StreamingMovies"])

le.fit(InternetService_order)
dff["NEW_InternetService_ORDERED_ENC"] = le.transform(dff["InternetService"])
le.fit(Contract_order)
dff["NEW_Contract_ORDERED_ENC"] = le.transform(dff["Contract"])
le.fit(tenure_qcut_order)
dff["NEW_tenure_qcut_ORDERED_ENC"] = le.transform(dff["NEW_TENURE_QCUT"])
le.fit(monthly_charges_order)
dff["NEW_monthly_charges_ORDERED_ENC"] = le.transform(dff["NEW_MONTHLY_CHARGES_CUT"])
dff.head( )
dff.shape
dff = dff.rename(columns={"Credit card (automatic)":"Credit_card_(automatic)",
            "Electronic check":"Electronic_check",
            "Mailed check":"Mailed_check"})
dff.columns

dff_ready = dff[["gender","SeniorCitizen","Partner",
                 "Dependents","tenure","PhoneService",
                 "PaperlessBilling","MonthlyCharges",
                 "new_total_charges_knn","NEW_CONTRACT_DURATION",
                 "NEW_MULTIPLELINES_ORDERED_ENC","NEW_OnlineSecurity_ORDERED_ENC",
                 "NEW_OnlineBackup_ORDERED_ENC","NEW_DeviceProtection_ORDERED_ENC",
                 "NEW_TechSupport_ORDERED_ENC","NEW_StreamingTV_ORDERED_ENC","NEW_StreamingMovies_ORDERED_ENC",
                 "NEW_InternetService_ORDERED_ENC","NEW_Contract_ORDERED_ENC","NEW_tenure_qcut_ORDERED_ENC",
                 "NEW_monthly_charges_ORDERED_ENC","Credit_card_(automatic)",
                 "Electronic_check","Mailed_check"]]
dff_ready.head()
dff_ready.info()
######################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
######################################################
scaler = RobustScaler()
dff_ready = pd.DataFrame(scaler.fit_transform(dff_ready),columns=dff_ready.columns)
dff_ready
######################################################
# Adım 5: Model oluşturunuz.
######################################################
le = LabelEncoder()
dff["Churn"] = le.fit_transform(dff["Churn"])

y = dff["Churn"]
X = dff_ready

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
