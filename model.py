#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data=pd.read_json('jsonfile',lines=True)


# In[2]:


#data.to_csv('data.csv')


# # Above codes are for creating csv from json

# In[1]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[2]:


data=pd.read_csv('data.csv')


# ## Preprocessing

# In[3]:


ask_data=data[data['subreddit']=='AskReddit']
#Taking only those data which are from Subreddit AskReddit


# In[4]:


ask_data.head(10)


# In[5]:


ask_data.shape


# In[6]:


#Removing the contents which
ask_data=ask_data[ask_data['body']!='[deleted]']


# In[7]:


ask_data['body']


# # Preprocessing

# In[8]:


ask_data


# In[9]:


ask_data.info()


# ### Observations:
# 1. Feature 'removal_reason' is all null. So it can be removed. <br>
# 2. Feature 'distinguished' has only two non-null. So it can be removed <br>
# 3. Feature 'retrieved_on' is actually a datetime column. So I will keep it on hold and check whether in its absence, my model works well or not.
# 4. Feature 'gilded' is all 0. There is no next number. So it can be removed from the column. It is of no use.<br>
# 5. Feature 'downs' is all 0. There is no next number. So it can be removed from the column. It is of no use.<br>
# 6. Feature 'controversiality' has only one value 1 and others 0. It is also like an outlier. So it is also of no use.<br>
# 7. Feature 'score_hidden' has all values False. So it can be removed from the column. It is of no use
# 9. Since created_utc is a datetime column, it doesnt hold any importance.<br>
# 10. The ups column is our target's copy. I will remove it.<br>
# 11. Feature 'Unnamed 0' is just the serial number which is useless.<br>
# 12. Feature 'author_flair_css_class' is all null. So it can be removed.
# 13. Feature 'subreddit' contains all same entry. It can be removed.
# 14. The id column is all unique. So it does not contribute at all to our data. It can be removed
# 15. The feature 'subreddit_id' has only one value all over. So it is also redundant.
# 16. The feature 'author_flair_text' is all empty column. It can also be removed.
# 17. Feature 'parent_id' is just random name. It does not play role.
# 18. The feature 'name' is all unique, does not contribute either.
# 19. Feature 'archived' has all True values. Can be removed.
# ### Features that can be used (guessed) are
# -link id <br>
# -author

# In[10]:


ask_data=ask_data.drop(columns=['removal_reason','distinguished','retrieved_on','gilded','downs','controversiality','score_hidden','created_utc','ups','Unnamed: 0','author_flair_css_class','subreddit','id','subreddit_id','author_flair_text','name','archived'])


# <b>Note- The features like parent_id and link_id may play the role in modeling but we cant ask such input from the user. So removing them.</b>

# In[11]:


ask_data=ask_data.drop(columns=['link_id','parent_id','author'])


# In[12]:


ask_data.head()


# In[13]:


ask_data.info()


# 20. The feature 'score' is actually a label. So we will remove it.

# In[14]:


#For now, we will just take comments as x and score as our y value
y=ask_data['score']
ask_data=ask_data.drop(columns='score',axis=1)


# In[15]:


ask_data


# In[16]:


x=ask_data['body'].values


# ### Cleaning the Text

# In[17]:


x


# In[18]:


def cleaner(data):
    data1=re.sub('[^a-zA-Z]',' ',data)
    data2=data1.lower()
    data3=data2.strip()
    data4=nltk.word_tokenize(data3)
    data5=[i for i in data4 if i not in set(stopwords.words('english'))]
    data6=[WordNetLemmatizer().lemmatize(i) for i in data5]
    data7=' '.join(data6)
    return data7
cleaner('hi How! are 34 the biggest psychological nicest historians')


# In[19]:


for k in range(len(x)):
    x[k]=cleaner(x[k])


# In[20]:




# In[21]:


len(x)


# In[22]:


y.shape


# ## Modeling

# ### Using Doc2Vec 
# 

# <b>Word2Vec creates sentiments with words where Doc2Vec creates sentiments with sentences.</b>
# 

# In[23]:


import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# In[24]:


tagged_data=[TaggedDocument(d,[i]) for i,d in enumerate(x)]


# In[25]:


len(tagged_data)


# In[26]:


model=Doc2Vec(tagged_data,vector_size=20,epochs=100,min_count=0)

#Save this doc2vec model
pickle.dump(model,open('docvec.pkl','wb'))

# In[27]:


model.corpus_count


# In[28]:


o_model=model.docvecs.vectors_docs


# In[29]:


o_model.shape


# In[30]:


model.docvecs.vectors_docs.shape


# In[31]:


y.shape


# ### Adding 'edited' feature to the model

# In[32]:


val=ask_data['edited'].values


# In[33]:


newpd=pd.DataFrame(o_model)


# In[34]:


newpd.insert(loc=20,column=20,value=val)


# In[35]:


final_op=newpd.values


# In[36]:


final_op.shape


# In[54]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
final_op_scaled=ss.fit_transform(final_op)


# 
# ## Train Test Split

# In[59]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(final_op_scaled,y,test_size=0.2)


# ## Trying ElasticNet

# In[60]:


from sklearn.linear_model import ElasticNet


# In[61]:


elr=ElasticNet()
elr.fit(xtr,ytr)








# In[63]:


y_pred=elr.predict(xte)


# In[64]:



# In[65]:


yte


# In[66]:


from sklearn.metrics import mean_squared_error,r2_score
print(np.sqrt(mean_squared_error(yte,y_pred)))
print(r2_score(yte,y_pred))


# ### Checking with one sentence as input



#Checking with input sentence


def for_one_input(a,b):
    cleaned_inp=cleaner(a)
    listed=cleaned_inp.split()
    vectorized_inp=model.infer_vector(listed)
    all_added=np.append(vectorized_inp,b)
    all_added_scaled=ss.fit_transform(all_added.reshape(-1,1))
    return elr.predict(all_added_scaled.reshape(1,-1)),0



#Checking with input sentence
inp_sent=input('Enter the sentence: ')
inp_edit=int(input('Enter the number of edits made:'))

print('The number of comments you will receive is ,',for_one_input(inp_sent,inp_edit))


# ## Deployment


#Saving it as pickle file
import pickle


pickle.dump(elr,open('elr.pkl','wb'))




#trying to load it
elr_loaded=pickle.load(open('elr.pkl','rb'))




'''

# # THE CODES BELOW THIS BLOCK ARE RUDIMENTARY😜

# ## Trying NN 

# ### NN Preprocessing

# In[28]:


import tensorflow 
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[29]:


tk=Tokenizer(num_words=10000,oov_token='OOV')
tk.fit_on_texts(x)


# In[32]:


seq=tk.texts_to_sequences(x)


# In[45]:


padded_seq=keras.preprocessing.sequence.pad_sequences(seq,maxlen=100)


# In[52]:


max_len=max([len(i.split()) for i in x])
vocab_size=len(tk.word_index)+1
print(max_len,vocab_size)


# ### NN Modeling

# In[58]:


model=keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=vocab_size,output_dim=50,input_length=max_len))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(50)))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(1))


# In[59]:


model.summary()


# In[60]:


model.compile(loss='mean_squared_error',optimizer='adam',metrics='accuracy')


# In[63]:


hist=model.fit(padded_seq,y,batch_size=32,epochs=10,validation_split=0.3)


# ### Using TFIDF

# In[50]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
x_vec=tfidf.fit_transform(x)


# ## Creating the model

# In[54]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x_vec,y,test_size=0.3,random_state=12)


# In[55]:


print(xtr.shape,ytr.shape,xte.shape,yte.shape)


# ### ExtraTreesRegressor

# In[125]:


from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor(n_estimators=100,max_depth=5)


# In[126]:


etr.fit(xtr,ytr)


# In[127]:


y_bar=etr.predict(xte)


# In[141]:


yte=yte.values


# In[148]:


yte[:20]


# In[152]:


y_bar[:20]


# ### LinearRegression

# In[56]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtr,ytr)


# In[58]:


y_bar=lr.predict(xte)


# In[60]:


yte[:20]


# In[61]:


y_bar[:20]

'''