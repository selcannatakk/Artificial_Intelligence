# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:35:21 2022

@author: selca
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# uyarıları kapatma
import warnings

warnings.filterwarnings("ignore")

# veriyi içeri aktarma
veri = pd.read_csv("olimpiyatlar/athlete_events.csv")
veri.head()
# veri hakkında genel bilgi
veri.info()

# verinin temizlenmesi
veri.columns
print(veri.columns)

#sütun ismini değiştirme 
veri.rename(columns={'ID' : 'id',
                    'Name' : 'isim',
                    'Sex' : 'cinsiyet',
                    'Age' : 'yas',
                    'Height' : 'boy',
                    'Weight' : 'kilo',
                    'Team' : 'takim',
                    'NOC' : 'uok',
                    'Games' : 'oyunlar',
                    'Year' : 'yil',
                    'Season' : 'sezon',
                    'City' : 'sehir',
                    'Sport' : 'spor',
                    'Event' : 'etkinlik',
                    'Medal' : 'madalya', 
                    },inplace = True)
print(veri.columns)
veri.head()

#yararsız verileri çıkarmak (drop)

veri = veri.drop(["id", "oyunlar"], axis = 1)#axis = 1 -> 1 sütun manasında
veri.head()

# kayıp veri yani nan olan veriler -> biz suan ort gore dolduruyoruz
# boy ve kilo için yapacağız

essiz_etkinlik = pd.unique(veri.etkinlik)# etkinlik ort gore doldurucaz
print("eşsiz etkinlik sayısıs: {}".format(len(essiz_etkinlik)))

# her bır etkınlıgı itaretif olarak dolas
# etkinlik özelinde ıle boy ve kılo ort hesapla
# etkinlik özelinde ıle boy ve kılo  değerlerini etkinlik ortalamarına eşitle

veri_gecici = veri.copy() # gercek verıyı bozmamak ıcın

boy_kilo_liste = ["boy", "kilo"]

for e in essiz_etkinlik:
    #etkinlik filtresi oluşturalmım
    etkinlik_filtre = veri_gecici.etkinlik == e
    #veriyi etkinliğe göre filtreleme
    veri_filtreli = veri_gecici[etkinlik_filtre]
    
    #boy ve kilo için etkinlik ort heszaplama
    for s in boy_kilo_liste:
        #np.round fonk ->ortlamayı yuvarla
        ortalama = np.round(np.mean(veri_filtreli[s]),2)
        if ~np.isnan(ortalama): #eğer etkinlik özelinde ort varsa 
            veri_filtreli[s] = veri_filtreli[s].fillna(ortalama)
        else:#eğer etkinlik özelinde ort varsa ortalamayı hesapla
            tum_veri_ortalamasi = np.round(np.mean(veri[s]),2)
            veri_filtreli[s] = veri_filtreli[s].fillna(tum_veri_ortalamasi)
    #etkinlik özelinde kayıp değerleri doldurulmus olan verıyı veri geciicye eşitle
    veri_gecici[etkinlik_filtre] = veri_filtreli
    
#giderilmiş olan gecici veriyi gercek veriye eşitle
veri = veri_gecici.copy()
veri.info() # boy ve kilo sütunlarının kayıp deger sayısı

# Yas sütunu kayıp verileri doldurma
#yas değişkeninde tanımlı olmayanları bulma

yas_ortalamasi = np.round(np.mean(veri.yas),2)
print("yas ortalaması: {}".format(yas_ortalamasi))
veri["yas"] = veri["yas"].fillna(yas_ortalamasi)
veri.info()

#madalya alamayan sporcuları verı setınden cıkarma
madalya_degiskeni = veri["madalya"]
madalyasi_null_olanlar = pd.isnull(madalya_degiskeni).sum()
print("madalyasi olmayanlar {}".format(madalyasi_null_olanlar))

# madalyası olanları ıstıyorum
madalya_degiskeni_filtresi = ~pd.isnull(madalya_degiskeni)

veri = veri[madalya_degiskeni_filtresi]
print(veri.head(5))

veri.info()

#sonradan kulllanmak için veriyi kaydediyoruz
veri.to_csv("olimpiyatlar/athlete_events_temizlenmis.csv", index = False)
 
# histogram grafıgı fonk 
def plotHistogram(degisken):
    """
       girdi: degisken/sütun ismi
       çıktı: ilgili degiskenin histokramı
        
     """
    plt.figure()
    plt.hist(veri[degisken], bins = 85, color = "orange" ) # 85 aralık
    plt.xlabel(degisken)
    plt.ylabel("Frekans")
    plt.title("Veri sıklığı - {}".format(degisken))
    plt.show()

sayisal_degisken = ["yas", "boy", "kilo","yil"]
for i in sayisal_degisken:
    plotHistogram(i)

veri.describe()

#kutu grafıgı
plt.boxplot(veri.yas)
plt.title("Yas Değişkeni İçin Kutu Grafiği")
plt.xlabel("yas")
plt.ylabel("değer")
plt.show()

#Katogorik değişkenler ile işlemler
#çubuk grafiği çizme
def plotBar(degisken, n=5):
    """
        girdi: degisken/sütun ismi
            n = gösterilecek essiz deger sayısı
            
        cıktı:çucuk grafıgı
    """
    
    veri_ = veri[degisken]
    veri_sayma = veri_.value_counts()
    veri_sayma = veri_sayma[:n]
    plt.figure()
    plt.bar(veri_sayma.index, veri_sayma, color = "orange" )
    plt.xticks(veri_sayma.index, veri_sayma.index.values)
    plt.xticks(rotation = 45)
    plt.xlabel("Frekans")
    plt.ylabel("Veri sıklıgı - {}".format(degisken))
    plt.show()
    print("{}: \n {}".format(degisken, veri_sayma))
    
kategorik_degisken = ["isim", "cinsiyet", "takim","uok","sezon","sehir","spor","etkinlik","madalya"]
for i in kategorik_degisken:
    plotBar(i)

#cinsiyete gore boy agırlık karsılastırma
erkek = veri[veri.cinsiyet == "M"]
erkek.head(3)
print(erkek.head(3))



































