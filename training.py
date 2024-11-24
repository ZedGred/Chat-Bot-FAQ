# prompt: buatkan kode untuk model faq interaktif untuk kebutuhan chatbot


# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Define FAQ data (replace with your own data)

data = pd.read_csv("data/chat.csv")



# Preprocess the text data
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("indonesian"))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]
    return " ".join(filtered_tokens)

question = data.question.to_numpy()
questions = [preprocess_text(question) for question in question]
answers = data.answer.to_numpy()


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "model/tfidf_matrix.pkl")
joblib.dump(questions, "model/questions.pkl")
joblib.dump(answers, "model/answers.pkl")


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


# Example usage
while True:
    user_input = input("Anda: ")
    if user_input.lower() == "keluar":
        break
    response = get_answer(user_input)
    print("Chatbot:", response)
