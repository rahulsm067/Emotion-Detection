import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import speech_recognition as sr
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST
import neattext.functions as nfx  # Added import for nfx
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Added import for pad_sequences

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))
pipe_svm = joblib.load(open("./models/emotion_classifier_pipe_svm.pkl", "rb"))

# Load LSTM model components
try:
    from tensorflow.keras.models import load_model
    lstm_model = load_model('./models/emotion_lstm_model.h5')
    lstm_tokenizer = joblib.load('./models/emotion_tokenizer.pkl')
    lstm_label_encoder = joblib.load('./models/emotion_label_encoder.pkl')
    LSTM_LOADED = True
except Exception as e:
    st.warning(f"Could not load LSTM model: {str(e)}")
    LSTM_LOADED = False

# Function to predict with LSTM
def predict_emotions_lstm(docx):
    if not LSTM_LOADED:
        return "LSTM model not available", np.zeros(len(lstm_label_encoder.classes_))
    
    # Preprocess
    text = nfx.remove_userhandles(docx)
    text = nfx.remove_stopwords(text)
    
    # Tokenize and pad
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    
    # Predict
    prediction = lstm_model.predict(padded)
    predicted_label = lstm_label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0], prediction[0]

# Functions for other models
def predict_emotions(docx, model_type='lr'):
    if model_type == 'lr':
        results = pipe_lr.predict([docx])
        return results[0]
    elif model_type == 'svm':
        results = pipe_svm.predict([docx])
        return results[0]

def get_prediction_proba(docx, model_type='lr'):
    if model_type == 'lr':
        results = pipe_lr.predict_proba([docx])
        return results
    elif model_type == 'svm':
        results = pipe_svm.predict_proba([docx])
        return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
                      "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
                      "shame": "😳", "surprise": "😮"}

def get_audio_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    
    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Emotion Detection in Text")
        
        # Model selection - only show LSTM if it loaded successfully
        model_options = ['Logistic Regression', 'SVM']
        if LSTM_LOADED:
            model_options.append('LSTM')
            
        model_type = st.radio(
            "Select Model:",
            model_options,
            horizontal=True
        )
        
        model_map = {
            'Logistic Regression': 'lr',
            'SVM': 'svm',
            'LSTM': 'lstm'
        }
        selected_model = model_map[model_type]

        # Add a button for voice input
        if st.button("🎤 Use Voice Input"):
            voice_text = get_audio_input()
            if voice_text:
                st.session_state.voice_text = voice_text

        with st.form(key='emotion_clf_form'):
            # Pre-fill the text area if voice input was used
            default_text = st.session_state.get('voice_text', '')
            raw_text = st.text_area("Type Here", value=default_text)
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            if selected_model == 'lstm':
                with st.spinner('Analyzing with LSTM...'):
                    prediction, probability = predict_emotions_lstm(raw_text)
                proba_array = probability.reshape(1, -1)  # Reshape for consistency
            else:
                prediction = predict_emotions(raw_text, selected_model)
                probability = get_prediction_proba(raw_text, selected_model)
                proba_array = probability

            add_prediction_details(raw_text, prediction, np.max(proba_array), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence: {:.2f}%".format(np.max(proba_array)*100))

            with col2:
                st.success("Prediction Probability")
                if selected_model == 'lstm':
                    classes = lstm_label_encoder.classes_
                else:
                    classes = pipe_lr.classes_ if selected_model == 'lr' else pipe_svm.classes_
                
                proba_df = pd.DataFrame(proba_array, columns=classes)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', 
                    y='probability', 
                    color='emotions',
                    tooltip=['emotions', 'probability']
                )
                st.altair_chart(fig, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))
        st.subheader("About the Emotion Detection App")
        st.write("""
    This intelligent emotion detection application leverages cutting-edge machine learning technologies 
    to analyze and classify emotional content in textual input. The system incorporates multiple 
    algorithmic approaches to deliver comprehensive sentiment analysis.
    """)
    
        st.subheader("Advanced Model Architecture")
        st.write("""
    Our platform employs an ensemble of sophisticated machine learning models to ensure accurate 
    emotion classification:
    - **Logistic Regression**: Provides fast and interpretable baseline predictions
    - **Support Vector Machines (SVM)**: Delivers high-accuracy classification with robust performance
    - **LSTM Neural Network**: Offers deep learning capabilities for complex emotional pattern recognition
    """)
    
        st.subheader("Key Features and Capabilities")
        st.markdown("""
    - **Multi-Model Analysis**: Choose between different machine learning approaches based on your needs
    - **Real-Time Processing**: Instant emotional sentiment detection with live results
    - **Confidence Metrics**: Transparent probability scores for each prediction
    - **Voice Input Support**: Optional speech-to-text functionality for hands-free operation
    """)
    
        st.subheader("Practical Applications")
        st.write("""
    This technology serves numerous real-world applications across industries:
    - Customer experience analysis and feedback interpretation
    - Social media sentiment monitoring and brand perception tracking
    - Psychological research and emotional pattern recognition
    - Content optimization for emotional impact and engagement
    - Market research and consumer behavior analysis
    """)
    
        st.subheader("Technical Implementation")
        st.write("""
    The application combines natural language processing techniques with machine learning algorithms, 
    implemented using Python's robust data science ecosystem including TensorFlow, scikit-learn, 
    and Streamlit for interactive web deployment.
    """)

if __name__ == '__main__':
    main()