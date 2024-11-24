# prompt: buatkan versi fast api untuk model di atas apakah modelnya dapat disimpan
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For saving and loading the model

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# Load the model and vectorizer (if saved previously)
try:
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("model/tfidf_matrix.pkl")
    questions = joblib.load("model/questions.pkl")
    answers = joblib.load("model/answers.pkl")
except FileNotFoundError:
    # Error
    print("Error: Model not found. Please train the model first.")

def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        stop_words = set(stopwords.words('indonesian'))
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        return " ".join(filtered_tokens)

# Define a function to find the most relevant answer
def get_answer(user_query):
    processed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    best_match_index = similarity_scores.argmax()
    if similarity_scores[0][best_match_index] > 0.5:  # Adjust the threshold as needed
        return answers[best_match_index]
    else:
        return "Maaf, saya tidak mengerti pertanyaan Anda."


class Query(BaseModel):
    query: str

@app.post("/get_response/")
async def get_response(query: Query):
    response = get_answer(query.query)
    return {"response": response}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)