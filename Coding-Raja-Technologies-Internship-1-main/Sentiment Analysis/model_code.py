import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

combined_data = pd.read_csv('data.csv')




tfidf = TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_data['Cleaned_Tweets'].values.astype('U'))
tfidf_df = pd.DataFrame(tfidf_matrix.todense())


train_tfidf_matrix = tfidf_matrix[:31962]
train_tfidf_matrix.todense()

x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,combined_data[:31962]['label'],test_size=0.3,random_state=17)
log_Reg = LogisticRegression(random_state=0,solver='lbfgs')
log_Reg.fit(x_train_tfidf,y_train_tfidf)

pickle.dump(log_Reg,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))