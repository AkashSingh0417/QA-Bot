import nltk

# Download required NLTK packages
nltk.download('vader_lexicon')
nltk.download('punkt')  # For tokenization
nltk.download('averaged_perceptron_tagger')  # For POS tagging
nltk.download('maxent_ne_chunker')  # For named entity recognition
nltk.download('words')  # For word lists
print("NLTK packages downloaded successfully!") 