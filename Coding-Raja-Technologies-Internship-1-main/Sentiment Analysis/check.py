import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

model=pickle.load(open("model.pkl","rb"))
combined_data = pd.read_csv('data.csv')
text="hate"
tfidf = TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_data['Cleaned_Tweets'].values.astype('U'))
tfidf_df = pd.DataFrame(tfidf_matrix.todense()) 
test = tfidf.transform([text])
pred_val=model.predict_proba(test)
print(pred_val)