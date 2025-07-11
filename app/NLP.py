 import streamlit as st
 import joblib
 import re
 import nltk
 import spacy
 from nltk.corpus import stopwords
 from nltk.tokenize import word_tokenize

 try:
     # Exemple de chargement du modèle de Régression Logistique et du vectoriseur TF-IDF
     model = joblib.load('best_sentiment_model.pkl') # Par exemple, lr_model
     vectorizer = joblib.load('tfidf_vectorizer.pkl') # Par exemple, tfidf_vectorizer
     nlp_spacy = spacy.load("fr_core_news_sm")
     stop_words_french = set(stopwords.words('french'))
 except FileNotFoundError:
     pass

def preprocess_text_for_prediction(text):
     text = text.lower()
     text = re.sub(r'[^a-zÀ-ÿ\s]', '', text)
     tokens = word_tokenize(text, language='french')
     filtered_tokens = [word for word in tokens if word not in stop_words_french and len(word) > 1]
     doc = nlp_spacy(" ".join(filtered_tokens))
     lemmas = [token.lemma_ for token in doc]
     return " ".join(lemmas)

st.title("Prédicteur de Sentiment d'Avis Clients Français")
st.write("Entrez un avis client en français pour prédire son sentiment (Positif/Négatif).")

user_input = st.text_area("Entrez votre avis ici :", "Ce produit est fantastique, je l'adore!")

if st.button("Prédire le Sentiment"):
    if user_input:
        processed_input = preprocess_text_for_prediction(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction_proba = model.predict_proba(vectorized_input)[0] # Obtenir les probabilités
        prediction = model.predict(vectorized_input)[0]

        sentiment_label = "Positif" if prediction == 1 else "Négatif"
        # La probabilité affichée est celle de la classe prédite.
        # Si la classe prédite est 1 (Positif), on prend prediction_proba[1].
        # Si la classe prédite est 0 (Négatif), on prend prediction_proba[0].
        predicted_confidence = prediction_proba[prediction]

        st.subheader("Prédiction :")
        st.write(f"**Sentiment :** {sentiment_label}")
        st.write(f"**Confiance :** {predicted_confidence*100:.2f}%")

        if sentiment_label == "Positif":
            st.success("😊 Cet avis est prédit comme étant **Positif**.")
        else:
            st.error("😠 Cet avis est prédit comme étant **Négatif**.")
    else:
        st.warning("Veuillez entrer un avis pour obtenir une prédiction.")

# Pour sauvegarder votre modèle et votre vectoriseur dans le notebook principal, ajoutez ces lignes APRÈS l'entraînement :
# import joblib
# joblib.dump(lr_model, 'best_sentiment_model.pkl') # Ou svm_model, rf_model - choisissez le meilleur
# joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
