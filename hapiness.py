
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import string
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

predict_happ_train = pd.read_csv('train.csv')
predict_happ_test = pd.read_csv('test.csv')

def response(Response):
    if Response == 'not happy':
        return 0
    else:
        return 1


predict_happ_train['Description_len'] = predict_happ_train['Description'].apply(len)
predict_happ_train['numeric_response'] = predict_happ_train['Is_Response'].apply(response)

x_train,x_test,y_train,y_test = train_test_split(predict_happ_train['Description'],
                                                 predict_happ_train['numeric_response'],test_size = 0.2)
def text_preprocess(descr):
    nonpunc = [char for char in  descr if char not in string.punctuation ]
    nonpunc = ''.join(nonpunc)
    clean_descr = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    return clean_descr


pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_preprocess)),
                    ('tfidf',TfidfTransformer()),
                     ('scaler',MaxAbsScaler()),
                    ('classifiers',SGDClassifier(loss = 'hinge',
                                                 alpha = 1e-3,n_iter = 5,random_state = 777))])
pipeline.fit(x_train,y_train)
predictions = pipeline.predict(x_test)


from sklearn.metrics import accuracy_score
Print('ACCURACY')
print(accuracy_score(predictions,y_test))
