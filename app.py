import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()


def main():
    html_temp = """
<div style="background-color: #E6E6FA; padding: 16px; border-radius: 8px; box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.2);">
    <h2 style="color: #333333; text-align: center; font-size: 24px; margin-bottom: 16px; text-transform: uppercase;">Bank Customer Churn Prediction Using ML</h2>
</div>
"""


    
    st.markdown(html_temp, unsafe_allow_html=True)
    model = joblib.load('churn_predict_model')
    p1 = st.number_input("CreditScore")
    p2 = st.number_input("Age", 18, 90)
    p3 = st.slider("Tenure",0,15)
    p4 = st.number_input("Balance")
    p5 = st.slider("NumOfProducts",1,4)
    p6 = st.slider("HasCrCard",0,1)
    p7 = st.slider("IsActiveMember",0,1)
    p8 = st.number_input("EstimatedSalary")
    # Initialize default values
    Geography_Germany = 0
    Geography_Spain = 0
    Geography_France = 0
    p9 = st.slider("Geography",1,3) 
    if p9 == 1:
        Geography_Germany = 1
        Geography_Spain = 0
        Geography_France = 0
    elif p9 == 2:
        Geography_Germany = 0
        Geography_Spain = 1
        Geography_France = 0
    elif p9 == 3:
        Geography_Germany = 0
        Geography_Spain = 0
        Geography_France = 1
    p10 = st.slider("Gender : ", 0, 1)

    if st.button('Predict'):
        input_data = np.array([[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Germany, Geography_Spain, p10]])
        input_data_scaled = sc.fit_transform(input_data)
        pred = model.predict(input_data_scaled)
        
        st.balloons()
        if pred == 0:
            st.write("<h3 style='color:green;'>No Churn</h3>", unsafe_allow_html=True)
        else:
            st.write("<h3 style='color:yellow;'>Churn Detected</h3>", unsafe_allow_html=True)
    
        st.success('Thank You')


if __name__ == '__main__':
    main()