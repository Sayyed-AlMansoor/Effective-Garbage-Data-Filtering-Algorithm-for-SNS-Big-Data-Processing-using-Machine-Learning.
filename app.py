from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

# Load the model and vectorizer
with open('lgb_model.pkl', 'rb') as model_file:
    lgb_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the sentence tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Best threshold determined from the validation set
best_threshold = 0.84  # Replace this with the actual best threshold

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-zA-Z\s]', '', text, re.I|re.A)
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form['raw_data']
    
    # Split the input data into sentences
    sentences = tokenizer.tokenize(data)
    
    # Preprocess each sentence
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    print("Preprocessed Sentences:", preprocessed_sentences)
    
    # Vectorize the sentences
    data_vectorized = vectorizer.transform(preprocessed_sentences)
    
    # Predict using the model
    predictions = lgb_model.predict(data_vectorized)
    print("Raw Predictions:", predictions)
    predictions_thresholded = np.where(predictions > best_threshold, 1, 0)
    print("Thresholded Predictions:", predictions_thresholded)
    
    # Collect non-garbage sentences
    non_garbage_sentences = [sentence for sentence, pred in zip(sentences, predictions_thresholded) if pred == 0]
    
    # Return the non-garbage sentences
    return jsonify({'filtered_sentences': non_garbage_sentences})

if __name__ == '__main__':
    app.run(debug=True)
