from flask import Flask, render_template, request
import json
import pickle
from sklearn.linear_model import LogisticRegression
from fonctions import prediction, entrainement



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')

""" @app.route("/prediction",methods=['POST'])
def text():
    user_text = request.form.get('input_text')
    print(user_text)
    if len(user_text) > 10:
        user_text = "Positif"
    else :
        user_text = "Négatif"
    return json.dumps({'text_user':user_text})
     """
@app.route("/prediction",methods=['POST'])
def text():
    user_text1 = request.form.get('input_text1')
    user_text2 = request.form.get('input_text2')
    user_text3 = request.form.get('input_text3')
    user_text4 = request.form.get('input_text4')
    user_text5 = request.form.get('input_text5')
    user_text6 = request.form.get('input_text6')
    user_text7 = request.form.get('input_text7')
    user_text8 = request.form.get('input_text8')
    user_text9 = request.form.get('input_text9')
    user_text10 = request.form.get('input_text10')
    user_text11= request.form.get('input_text11')
    user_text12= request.form.get('input_text12')
    user_text13= request.form.get('input_text13')
    user_text14= request.form.get('input_text14')
    
    user_text=[user_text1, user_text2, user_text3, user_text4, user_text5, user_text6, user_text7, user_text8, user_text9,user_text10, user_text11, user_text12, user_text13, user_text14]
    retour=prediction(user_text)
    #return json.dumps({'text_user':retour})
    return render_template("interface.html", input_text=user_text,prediction=retour)




@app.route("/entrainement",methods=['GET'])
def entr(usertexte=None):
    retour = entrainement()
    #return json.dumps({'text_user':retour})
    return render_template("entrainement.html",entrainement=retour)


if __name__ == "__main__":
    app.run(debug=True)

