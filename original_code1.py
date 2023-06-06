import joblib
import pandas as pd
import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector



@st.cache_resource
def load_artifacts():
    model = joblib.load("model_flower.pickle")
    X_test = pd.read_csv("X.csv")
    y_test = pd.read_csv("y.csv")
    return model, X_test, y_test


def main():
    st.title("Iris Dataset ML Application")
    model, X_test, _ = load_artifacts()
    
    

    # script how user will unput his flower dimensions
    st.sidebar.header("Insert dimension of flower in centimeter you have got")
    
    def user_input_feature():
        Sepal_Length = st.sidebar.number_input('Sepal_Length between 0 cm to 10 cm', min_value=0.0, max_value=10.0, step=0.1, key = "SL")
        st.write('The Sepal_Length in cm is ', Sepal_Length)
        Sepal_Width = st.sidebar.number_input('Sepal_Width between 0 cm to 10 cm', min_value=0.0, max_value=10.0, step=0.1, key = "SW")
        st.write('The Sepal_Width in cm is ', Sepal_Width)
        Petal_Length = st.sidebar.number_input('Petal_Length between 0 cm to 10 cm', min_value=0.0, max_value=10.0, step=0.1,key = "PL")
        st.write('The Petal_Length in cm is ', Petal_Length)
        Petal_Width = st.sidebar.number_input('Sepal_Width between 0 cm to 10 cm', min_value=0.0, max_value=10.0, step=0.1,key = "PW")
        st.write('The Petal_Width in cm is ', Petal_Width)
        
        
        
        data = {'Sepal_Length': Sepal_Length, 'Sepal_Width': Sepal_Width, 
                'Petal_Length':Petal_Length, 'Petal_Width': Petal_Width}
        features = pd.DataFrame(data, index =[0])
        
        return features
    
    df = user_input_feature()
    
    #printing specified input parameter for the model
    st.header("Dimension of flower you have got are")
    st.write(df)
    p = model.predict(df.iloc[:1])[0]
    
    flower_type = {0 :"Iris Setosa", 1 : "Iris Versicolour", 2 : "Iris Virginica" }
    
    st.write("As per model's prediction your flower is of type", flower_type[p])
    
    # using feedback colletor
    collector = FeedbackCollector()
    # collecting feedback here
    q1 = st.text_input("Q 1 Do you think your your flower type is same as predicted by model? (YES/N0)")
    q2 = st.text_input("Q 2 If 'NO' what  do you think your flower type is? (It is of same type/Iris Setosa/Iris Versicolour/Iris Virginica)")
    q3 = st.text_input("Q 3 Are you happy with model's prediction ? (YES/N0)")
    slider = st.slider("Q 4 How lokely would you recommend the model to people in your network?", max_value=10, value=9)
    submit = st.button("Save feedback")
    if q1 and q2 and q3 and slider and submit:
        feedback = collector.st_feedback(
            "custom",
            user_response={
                "Q 1": q1,
                "Q 2": q2,
                "Q 3": q3,
                "Q 4" : slider
            },
            path= "./collected_feedback.json"
        )
        feedback.dict() if feedback else None        
    
    
if __name__ == '__main__':
    main()
