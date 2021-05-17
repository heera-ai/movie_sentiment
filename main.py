import streamlit as st
from PIL import Image
from sklearn.externals import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

# from model import text_cleaner,predict_on_ip
def text_cleaner(message):
    rm_punctuation = "".join([text for text in message if text not in ["!",".","$","#","(",")","{","}","[","]","'",'"',",","`"]])
    word_tokens = word_tokenize(rm_punctuation)
    clean_text = " ".join(w for w in word_tokens if not w in stop_words)
    return clean_text

def predict_on_ip(usr_ip):
	usr_ip = text_cleaner(usr_ip)
	review_vec = count_vect.transform([usr_ip])
	result = {1:"Positive",0:"Negative"}
	return result[model.predict(review_vec)[0]]


model = joblib.load("trained_model.pkl")
count_vect = joblib.load("vectors.pkl")

nav= st.sidebar.radio("Navigations",["Home","Data","Model","Code","Contact Us"],index=0)

st.sidebar.write("""Contact us \n
heeralallegha@gmail.com \n
+91-9649488704""")

if nav == "Home":
	st.title('Movie Review Sentiment Analysis!!')
	img = Image.open('movie_img.jpg')
	st.image(img)
	st.header("""Write your review about the movie and predict the Sentiment""")
	user_input = st.text_area("Please enter your movie review here")
	def predict():
		if len(user_input) == 0:
			return st.error("Please enter a valid review")
		prediction = predict_on_ip(user_input)
		st.write("Your review's sentiment is : ",prediction)
		return
	if st.button("Predict"):
		predict()


if nav == "Data":
	st.title("Read more about Data")
	st.write("## Data Source","Kaggle.com")

if nav == "Model":
	st.header("Modle Training")

if nav == "Code":
	st.title("Python code for model training")

if nav == "Contact Us":
	st.title("Wellcome to the world of Predictions")
	st.subheader("Please feel free to write us about your experience with us")

