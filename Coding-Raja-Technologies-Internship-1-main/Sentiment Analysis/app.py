from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

model=pickle.load(open("model.pkl","rb"))
app=Flask(__name__,template_folder='template')
@app.route('/')
def hello_world():
    return render_template('web.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        text=request.form.get("info")
        combined_data = pd.read_csv('data.csv')
        tfidf = TfidfVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
        tfidf_matrix = tfidf.fit_transform(combined_data['Cleaned_Tweets'].values.astype('U'))
        tfidf_df = pd.DataFrame(tfidf_matrix.todense()) 
        test = tfidf.transform([text])
        pred_val=model.predict_proba(test)
        
        if(pred_val[:,0]>=0.75):
            return render_template("web.html",pred="Sentiment Of Text Is "+"POSITIVE")
        elif(pred_val[:,0]>0.4):
            return render_template("web.html",pred="Sentiment Of Text Is "+"NEUTRAL")
        else:
            return render_template("web.html",pred="Sentiment Of Text Is "+"NEGATIVE")

if __name__ == '__main__':
    app.run(debug=True)