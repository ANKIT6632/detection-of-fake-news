
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv("news.csv")
data.shape
data.head(10)
labels=data.label
x_train,x_test,y_train,y_test=train_test_split(data['text'], labels, test_size=0.3, random_state=7)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
model1=PassiveAggressiveClassifier(max_iter=50)
model1.fit(tfidf_train,y_train)

pickle.dump(model1,open('model.pk1','wb'))
model=pickle.load(open('model.pk1','rb'))

y_pred=model.predict(tfidf_test)


app=Flask(__name__)
model=pickle.load(open('model.pk1','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	msg=request.form.get('check1')
	#df=pd.DataFrame([str(msg)],columns=['label'])
	df=([msg])
	msg1=tfidf_vectorizer.transform(df)
	prediction=model.predict(msg1)

	if prediction==['FAKE']:
		return render_template('fake.html',prediction_text="the given news is {}".format(msg))
	else:
		return render_template('real.html',prediction_text="the given news is {}".format(msg))

    #return render_template('index.html',prediction_text="the given news is {}".format(prediction))

if __name__=="__main__":
	app.run(debug=True)