# Keep these imports as-is
from flask import Flask, request, jsonify, render_template
import assemblyai as aai
import os
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename

# Set up NLTK data directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
aai.settings.api_key = "45a8ba05f6f14df7bb83cfd3fd5fdb1e"
sid = SentimentIntensityAnalyzer()

# Create uploads directory if not exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('flask.html')

def check_for_alerts(utterance, prev_utterance, response_time):
    """Check for potential issues in responses"""
    alerts = []
    
    # Check for very negative sentiment
    if utterance['sentiment'] < -0.5:
        alerts.append({
            'type': 'sentiment',
            'level': 'high',
            'message': f"Very negative response detected from Speaker {utterance['speaker']}"
        })
    
    # Check for slow response time
    if response_time and response_time > 5:  # if response takes more than 5 seconds
        alerts.append({
            'type': 'response_time',
            'level': 'medium',
            'message': f"Slow response time ({response_time:.1f}s) from Speaker {utterance['speaker']}"
        })
    
    # Check for very short responses (possibly dismissive)
    if len(utterance['text'].split()) < 3:
        alerts.append({
            'type': 'response_length',
            'level': 'medium',
            'message': f"Very short response from Speaker {utterance['speaker']}"
        })
    
    return alerts

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            file_path,
            config=aai.TranscriptionConfig(speaker_labels=True)
        )
        
        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({'error': transcript.error})
        
        analysis_data = []
        performance_metrics = {
            'response_times': [],
            'sentiments': [],
            'total_responses': 0,
            'avg_response_time': 0,
            'avg_sentiment': 0,
            'utterances': [],
            'alerts': []
        }
        
        prev_speaker_end = None
        prev_utterance = None
        
        for utterance in transcript.utterances:
            sentiment = sid.polarity_scores(utterance.text)
            entry = {
                'text': utterance.text,
                'speaker': utterance.speaker,
                'start': utterance.start,
                'end': utterance.end,
                'sentiment': sentiment['compound']
            }
            analysis_data.append(entry)
            performance_metrics['utterances'].append(entry)
            
            # Calculate response time and check for alerts
            response_time = None
            if prev_speaker_end and utterance.start:
                response_time = (utterance.start - prev_speaker_end) / 1000
                performance_metrics['response_times'].append(response_time)
            
            # Check for alerts if this is a response
            if prev_utterance and utterance.speaker != prev_utterance['speaker']:
                alerts = check_for_alerts(entry, prev_utterance, response_time)
                if alerts:
                    performance_metrics['alerts'].extend(alerts)
            
            if utterance.speaker == 'B':
                performance_metrics['total_responses'] += 1
            
            prev_speaker_end = utterance.end if utterance.end else prev_speaker_end
            prev_utterance = entry
        
        if performance_metrics['response_times']:
            performance_metrics['avg_response_time'] = np.mean(performance_metrics['response_times'])
        
        bot_sentiments = [u['sentiment'] for u in analysis_data if u['speaker'] == 'B']
        if bot_sentiments:
            performance_metrics['avg_sentiment'] = np.mean(bot_sentiments)
        
        return jsonify({
            'conversation': analysis_data,
            'metrics': performance_metrics,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)