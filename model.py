# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# import pickle
# from sklearn.externals import joblib

# df = pd.read_csv("movie_review.csv")
# df.drop(columns=["fold_id","cv_tag","html_id","sent_id"],axis=1,inplace = True)
# df["tag"] = df["tag"].map({"pos":1,"neg":0})
# stop_words = set(stopwords.words('english'))
# def text_cleaner(message):
#     rm_punctuation = "".join([text for text in message if text not in ["!",".","$","#","(",")","{","}","[","]","'",'"',",","`"]])
#     word_tokens = word_tokenize(rm_punctuation)
#     clean_text = " ".join(w for w in word_tokens if not w in stop_words)
#     return clean_text
    
# df["text"] = df["text"].apply(text_cleaner)
# X = df["text"]
# count_vect = CountVectorizer()
# X = count_vect.fit_transform(X)
# y = df["tag"]
# model = MultinomialNB()
# model.fit(X,y)

# joblib.dump(model, 'trained_model.pkl')
# joblib.dump(count_vect,'vectors.pkl')
# def predict_on_ip(usr_ip):
# 	usr_ip = text_cleaner(usr_ip)
# 	review_vec = count_vect.transform([usr_ip])
# 	result = {1:"Positive",0:"Negative"}
# 	return result[model.predict(review_vec)[0]]

# if __name__ == "__main__":
#     print("Model is ready!!!")

