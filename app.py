import packages.data_processor as dp
import streamlit as st
import joblib

# Load the model
spam_clf = joblib.load('./models/spam_detector_model.pkl')

# Load vectorizer
vectorizer = joblib.load('./vectors/vectorizer.pickle')


### MAIN FUNCTION ###
def main(title  = "Your Awesome Streamlit Text classification App".upper()):
    st.markdown("<h1 style='text-align:center; font-size: 65px; color: #4682B4;'>{}"
                "</h1>".format(title),
                unsafe_allow_html=True)
    st.image("./images/message-image.jpeg")
    info = ''

    with st.expander("1. Check if your text is a spam or ham :D"):
        text_message = st.text_input("Please enter your messgage")
        if st.button("Predict"):
            predicion = spam_clf.predict(vectorizer.transform([text_message]))

            if predicion[0] == 0:
                info = 'Ham'
            else:
                info = 'Spam'

            st.success('Prediction: {}'. format(info))


if __name__ == "__main__":
    main()
