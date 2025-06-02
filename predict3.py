import assemblyai as aai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def format_timestamp(milliseconds):
    """Convert milliseconds to a readable time format (MM:SS.mmm)"""
    if milliseconds is None:
        return "00:00.000"
    
    seconds = int(milliseconds / 1000)
    ms = milliseconds % 1000
    minutes = int(seconds / 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

# Configure AssemblyAI
aai.settings.api_key = "45a8ba05f6f14df7bb83cfd3fd5fdb1e"

# Knowledge base for accuracy checking
KNOWLEDGE_BASE = {
    'packages': ['package', 'days', 'price', 'accommodation'],
    'travel_info': ['climate', 'places', 'distance', 'transport'],
    'booking': ['website', 'book', 'payment', 'cancel']
}

def check_knowledge_coverage(user_text, bot_text, category):
    """Check if bot response matches knowledge base"""
    required_keywords = KNOWLEDGE_BASE[category]
    return (any(kw in user_text.lower() for kw in required_keywords) and
        any(kw in bot_text.lower() for kw in required_keywords))

def analyze_bot_performance(utterances):
    """Analyze bot responses with proper indentation"""
    analysis = {
        'total_responses': 0,
        'relevant_responses': 0,
        'correct_info': 0,
        'generic_responses': 0,
        'response_times': [],
        'similarity_scores': []
    }

    tfidf = TfidfVectorizer(stop_words='english')
    prev_user_utterance = None

    for i, utterance in enumerate(utterances):
        if utterance.speaker == 'A':
            prev_user_utterance = utterance
        elif utterance.speaker == 'B' and prev_user_utterance:
            analysis['total_responses'] += 1
            
            # Calculate response time
            if utterance.start and prev_user_utterance.end:
                response_time = (utterance.start - prev_user_utterance.end) / 1000
                analysis['response_times'].append(response_time)
            
            # Calculate similarity
            texts = [prev_user_utterance.text, utterance.text]
            tfidf_matrix = tfidf.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            analysis['similarity_scores'].append(similarity)
            
            analysis['relevant_responses'] += 1 if similarity > 0.4 else 0
            
            # Check generic responses
            generic_phrases = ['let me check', 'please wait', 'thank you']
            analysis['generic_responses'] += 1 if any(
                phrase in utterance.text.lower() for phrase in generic_phrases
            ) else 0
            
            # Check knowledge coverage
            analysis['correct_info'] += 1 if any(
                check_knowledge_coverage(prev_user_utterance.text, utterance.text, category)
                for category in KNOWLEDGE_BASE
            ) else 0

    # Calculate averages
    analysis['avg_response_time'] = np.mean(analysis['response_times']) if analysis['response_times'] else 0
    analysis['avg_similarity'] = np.mean(analysis['similarity_scores']) if analysis['similarity_scores'] else 0
    
    return analysis

def main():
    """Main execution flow"""
    print("Starting transcription...")
    
    # Transcribe audio file
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        "uploads/FBAI_Sample_English_India_CC_Travel2 (mp3cut.net).wav",
        config=aai.TranscriptionConfig(speaker_labels=True)
    )
    
    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription error: {transcript.error}")
        return
    
    print("\nConversation Transcript with Timestamps:")
    for utterance in transcript.utterances:
        start_time = format_timestamp(utterance.start)
        end_time = format_timestamp(utterance.end)
        print(f"[{start_time} -> {end_time}] Speaker {utterance.speaker}: {utterance.text}")
    
    # Analyze bot performance
    print("\nAnalyzing bot performance...")
    analysis = analyze_bot_performance(transcript.utterances)
    
    print("\n--- Bot Performance Report ---")
    print(f"Total Responses: {analysis['total_responses']}")
    print(f"Relevant Responses: {analysis['relevant_responses']} ({analysis['avg_similarity']:.2f} avg similarity)")
    print(f"Correct Information: {analysis['correct_info']}")
    print(f"Generic Responses: {analysis['generic_responses']}")
    print(f"Average Response Time: {analysis['avg_response_time']:.1f}s")

if __name__ == "__main__":
    main()
    print("\nAnalysis complete!")