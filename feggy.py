import pickle
import numpy as np
import streamlit as st
from PIL import Image

#load save model
rfc_100=pickle.load(open('RFC_model.pkl','rb'))
scale=pickle.load(open('scaler.pkl','rb'))

#judul web
st.title("prediksi pemain basket dengan RFC")
primaryColor="#F63366"

#untuk input data
col1, col2=st.columns(2)
with col1:
    GP=st.number_input("GP")
with col2:
    MIN=st.number_input("MIN")
with col1:
    PTS=st.number_input("PTS")
with col2:
    FGM=st.number_input("FGM")
with col2:
    FGA=st.number_input("FGA")
with col2:
    FG_percent=st.number_input("FG_percent")
with col2:
    FTM=st.number_input("FTM")
with col2:
    FTA=st.number_input("FTA")
with col2:
    FT_percent=st.number_input("FT_percent")
with col2:
    OREB=st.number_input("OREB")
with col2:
    DREB=st.number_input("DREB")
with col2:
    REB=st.number_input("REB")
with col2:
    AST=st.number_input("AST")
with col2:
    STL=st.number_input("STL")
with col2:
    BLK=st.number_input("BLK")
with col2:
    TOV=st.number_input("TOV")


#kode untuk predikisi
prediksi =''
if st.button("Prediksi SEKARANG"):
    # Mengubah argumen menjadi array numpy dua dimensi
    sc=scale.transform([[GP, MIN, PTS, FGM, FGA, FG_percent, FTM, FTA, FT_percent, OREB, DREB, REB, AST, STL, BLK, TOV]])
    # Melakukan prediksi dengan XGBoost
    Prediksi_Pemain = rfc_100.predict([[sc[0][0],sc[0][1],sc[0][2],sc[0][3],sc[0][4],sc[0][5],sc[0][6],sc[0][7],sc[0][8],sc[0][9],sc[0][10],sc[0][11],sc[0][12],sc[0][13],sc[0][14],sc[0][15]]])
    
    if Prediksi_Pemain[0]==0:
        prediksi ="tidak layak"
    elif Prediksi_Pemain[0] == 1:
        prediksi = "layak"
    else:
        prediksi = "tidak ditemukan jenis"

st.success(prediksi)

#teks
st.caption('Developer')