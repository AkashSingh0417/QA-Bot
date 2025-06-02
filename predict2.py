import assemblyai as aai
import pandas as pd
import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# --- Core Functions ---
def calculate_response_time(utterances):
    """Calculate average response time between speaker turns"""
    response_times = []
    prev_end = None
    current_speaker = None
    
    for utterance in utterances:
        if not current_speaker:
            current_speaker = utterance.speaker
            prev_end = utterance.end
            continue
            
        if utterance.speaker != current_speaker:
            # Handle potential None values in timing data
            if utterance.start and prev_end:
                response_time = (utterance.start - prev_end) / 1000  # ms to seconds
                response_times.append(response_time)
            current_speaker = utterance.speaker
            
        if utterance.end:
            prev_end = utterance.end
    
    # Handle empty case to avoid division by zero
    return sum(response_times)/len(response_times) if response_times else 0

def process_conversation(utterances, encoders):
    """Process conversation utterances into features"""
    # Sentence-level sentiment analysis
    sentence_sentiments = []
    conversation_text = []
    
    for utterance in utterances:
        # Analyze each sentence individually
        sentiment = sid.polarity_scores(utterance.text)
        sentence_sentiments.append(sentiment['compound'])
        conversation_text.append(utterance.text)
    
    # Aggregate sentiment scores
    avg_sentiment = np.mean(sentence_sentiments) if sentence_sentiments else 0
    overall_sentiment = 'neutral'
    if avg_sentiment <= -0.05:
        overall_sentiment = 'negative'
    elif avg_sentiment >= 0.05:
        overall_sentiment = 'positive'

    # Calculate response time with error handling
    try:
        avg_response_time_sec = calculate_response_time(utterances)
    except Exception as e:
        print(f"Error calculating response time: {e}")
        avg_response_time_sec = 0

    # Load known categories with fallbacks
    try:
        issue_categories = encoders['Issue_Category'].classes_
        customer_issues = encoders['Customer_Issue'].classes_
        tech_responses = encoders['Tech_Response'].classes_
        issue_statuses = encoders['Issue_Status_ohe'].categories_[0]
    except KeyError as e:
        print(f"Missing encoder: {e}")
        return pd.DataFrame()

    features = {
        'Customer_Issue': customer_issues[0] if customer_issues else 'Unknown',
        'Tech_Response': tech_responses[0] if tech_responses else 'Unknown',
        'Issue_Category': issue_categories[0] if issue_categories else 'Unknown',
        'Response_Time_Min': avg_response_time_sec / 60,
        'Issue_Status': issue_statuses[0] if issue_statuses else 'Unknown',
        'Sentiment': overall_sentiment
    }

    # Add your existing keyword detection logic here
    # ...

    return pd.DataFrame([features])

# ... [Keep preprocess_features and predict_conversation functions mostly same] ...

def predict_conversation(utterances):
    """Generate predictions from conversation utterances"""
    try:
        artifacts = joblib.load('preprocessing_artifacts.pkl')
        model = load_model(
            'tech_support_model.h5',
            custom_objects={'MeanSquaredError': MeanSquaredError()}
        )
    except Exception as e:
        print(f"Error loading model/artifacts: {e}")
        return None

    try:
        raw_df = process_conversation(utterances, artifacts['encoders'])
        if raw_df.empty:
            print("Error: Empty features DataFrame")
            return None
            
        processed_df = preprocess_features(raw_df, artifacts['encoders'])
        
        # Debug print to check dataframe shape
        print(f"Processed DataFrame shape: {processed_df.shape}")
        
        scaled_prediction = model.predict(processed_df)[0][0]
        
        # Rest of your prediction logic...
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # Transcribe audio
    aai.settings.api_key = "45a8ba05f6f14df7bb83cfd3fd5fdb1e"
    FILE_URL = r"C:\Users\hp\OneDrive\Desktop\hack0.7\uploads\FBAI_Sample_English_India_CC_Travel2 (mp3cut.net).wav"
    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL, config=aai.TranscriptionConfig(speaker_labels=True))
    
    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        # Print conversation with sentence-level sentiment
        for utterance in transcript.utterances:
            sentiment = sid.polarity_scores(utterance.text)
            print(f"Speaker {utterance.speaker} [Sentiment: {sentiment['compound']:.2f}]: {utterance.text}")
        
        # Generate predictions
        prediction = predict_conversation(transcript.utterances)
        
        if prediction:
            print("\n--- Analysis ---")
            print(f"Avg Response Time: {prediction['response_time']} mins")
            print(f"Overall Sentiment: {prediction['sentiment'].capitalize()}")
            print(f"Predicted Resolution: {prediction['resolution_time']} mins")
            if prediction['alert']:
                print(f"\n🚨 {prediction['alert']}")