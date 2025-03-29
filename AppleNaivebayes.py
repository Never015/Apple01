import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 📌 โหลดข้อมูล Apple Dataset
Apple = pd.read_csv("./data/Apple2.csv")
X = Apple.drop(columns=['Quality'])
y = Apple.Quality
appleQ = Apple.Quality


# 📌 สร้างและฝึกโมเดล Naïve Bayes
model = GaussianNB()
model.fit(X,y)

# 📌 สร้าง Web App ด้วย Streamlit
st.title("AppleQuality")
st.write("กรอกข้อมูล Apple แล้วให้โมเดลทำนาย Quality")

# 📌 สร้างอินพุตให้ผู้ใช้กรอกข้อมูล
##A1 = st.number_input("กรุณากรอกข้อมูล A_id",0,3999)
A2 = st.slider("กรุณาเลือกข้อมูล Size ",-7.15,6.41)
A3 = st.slider("กรุณาเลือกข้อมูล Weight ",-7.15,5.79)
A4 = st.slider("กรุณาเลือกข้อมูล Sweetness ",-6.89,6.37)
A5 = st.slider("กรุณาเลือกข้อมูล Crunchiness",-6.06,7.62)
A6 = st.slider("กรุณาเลือกข้อมูล Juiciness",-5.96,7.36)
A7 = st.slider("กรุณาเลือกข้อมูล Ripeness",-5.86,7.24)
A8 = st.slider("กรุณาเลือกข้อมูล Acidity",-7.01,7.4)


# 📌 ปุ่มกดเพื่อทำนาย
if st.button("🔍 Predict"):
    user_input = np.array([[A2,A3,A4,A5,A6,A7,A8]])
    prediction = model.predict(user_input)
    predicted_class = appleQ[prediction[0]]
    st.success(f"🌼 ผลลัพธ์: ของการเป็นโรคหัวใจ **{predicted_class}**")

    ##if(predicted_class == 1) : st.success(f"🌼 ผลลัพธ์: Good")
    ##else : st.success(f"🌼 ผลลัพธ์: bad")


