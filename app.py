
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()



def cleaner(data):
    data1=re.sub('[^a-zA-Z]',' ',data)
    data2=data1.lower()
    data3=data2.strip()
    
    return data3


elr=pickle.load(open('elr.pkl','rb'))
docvec=pickle.load(open('docvec.pkl','rb'))


def for_one_input(a,b):
    cleaned_inp=cleaner(a)
    listed=cleaned_inp.split()
    vectorized_inp=docvec.infer_vector(listed)
    print(vectorized_inp)
    all_added=np.append(vectorized_inp,b)
    all_added_scaled=ss.fit_transform(all_added.reshape(-1,1))
    return np.round(elr.predict(all_added_scaled.reshape(1,-1)),0)



for_one_input('the sun rises in the east',1)


app = Flask(__name__,)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_text = request.form.get("sentence")
    inp_number = request.form.get("edit_numbers")
    predicted_value=for_one_input(inp_text,inp_number)
    return render_template('index.html', values_out="The no of upvotes you're likely to get is {}".format(predicted_value))
    

if __name__ == "__main__":
    app.run()
