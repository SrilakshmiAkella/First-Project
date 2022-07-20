from flask import Flask, render_template, url_for, flash, redirect, request
from flask import session
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import exc
from sqlalchemy import create_engine
from sqlalchemy.orm.util import identity_key
from table import newUsers, Base
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#from IPython.display import Image
import graphviz
from sklearn import tree
#import pydotplus 
import pickle

import os
import sklearn
import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

app = Flask(__name__)


# Connect to database8e
engine = create_engine('sqlite:///userregistration.db',connect_args={'check_same_thread':False},echo=True)
Base.metadata.bind = engine
# Create session
DBSession = sessionmaker(bind=engine)
session = DBSession()

@app.route("/")
@app.route("/login", methods = ["GET","POST"])
def login():
    if request.method == 'POST':
        users = session.query(newUsers).all()
        for user in users:
            if (user.email == request.form['mail'] and user.password == request.form['pswd']):
                return redirect(url_for('userInputs'))
        flash("Please enter correct username and password!!!")
        return render_template('login.html')
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('login.html')

@app.route("/output")
def output():

    return render_template('output.html')   

@app.route("/forgotpswd/",methods = ["GET","POST"])
def forgotpswd():
    if request.method == 'POST':
        users = session.query(newUsers).all()
        for user in users:
            if (user.email == request.form['mail']):
                flash("Your Old password is :")
                flash(user.password)
            # else:
            #     flash("User not  registered yet ..plz register") 
            #     return render_template("Registration.html")   

        return render_template('login.html')
     
    return render_template('forgotpswd.html')    

@app.route("/register/", methods = ["GET","POST"])
def register():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['mail']
        password = request.form['pswd']
        phnum = request.form['pnum']
        print(request.form['fname'], request.form['lname'], request.form['mail'])
        if (fname == '' or lname == '' or email == '' or password == '' or phnum == ''):
            flash('Please enter all the fields')
            return render_template('Registration.html')
        elif (fname != '' or lname != '' or email != '' or password != '' or phnum != ''):
            try:
                users = newUsers(fname=request.form['fname'],
                    lname=request.form['lname'], 
                    email=request.form['mail'], 
                    password=request.form['pswd'],
                    phoneNo=request.form['pnum'])
                if (len(request.form['pnum']) > 10 or len(request.form['pnum']) < 10) :
                    flash("Invalid Phone Number")
                    return render_template('Registration.html')
                session.add(users)
                session.commit()
                flash('You have Signed In!!! Please Login In')
                return render_template('login.html')
            except:
                flash("Email Already Exists")
                return render_template('Registration.html')
    
    return render_template('Registration.html')

@app.route("/userinputs/", methods = ["GET","POST"])
def userInputs():
    if request.method == 'POST':
        if (request.form['age_value'] == '' or request.form['bp'] == '' or request.form['deliveryno'] == '' or
            request.form['typofdel'] == '' or request.form['hrtprb'] == ''):
            flash('Please enter all the fields')
            return redirect(url_for('userInputs'))

        else:
            age =  request.form['age_value']
            dno =  request.form['deliveryno']
            hrtprob =  request.form['hrtprb']
            bldpr = request.form['bp']
            bbypos = request.form['babypos']
            typeofdeli = request.form['typofdel']
            #print(typeofdeli)
            dno=dno.split('-')
            print(dno[1])
            #dno = dno[1]
            bldpr=bldpr.split('/')
            if(hrtprob == "Yes"):
                hrtprob = 1
            else:
                hrtprob = 0    

            if(((int(bldpr[0])) >= 70 and (int(bldpr[0])) <=100) or ((int(bldpr[1])) <= 72)):
                bldpr = 'low'
            elif(((int(bldpr[0])) >= 101 and (int(bldpr[0])) <=128) or ((int(bldpr[1])) >= 74 and (int(bldpr[1])) <= 88)):
                bldpr = 'normal'  
            elif(((int(bldpr[0])) >= 130 or ((int(bldpr[1])) >= 90))):
                bldpr = 'high'

             

            row = [age,dno[1],bldpr,hrtprob,bbypos]

            if(row[2] == 'low'):
                row[2] = 0 
            elif(row[2] == 'normal'):
                row[2] = 1
            elif(row[2] == 'high'):
                row[2] = 2 

            if(row[4] == 'anterior'):
                row[4] = 0 
            elif(row[4] == 'HEADUP'):
                row[4] = 1
            elif(row[4] == 'TRANSVERSALIE'):
                row[4] = 2     

            print(row)
            print(typeofdeli)
            col_names = ['Age', 'Deliveryno', 'bloodpressure', 'heartproblem', 'babyposition', 'ceasarean']
            
            data=pd.read_csv('C://Users//DELL//Desktop/data/ceasarean.csv',header=None, names=col_names)    
            
            data['ceasarean'] = data['ceasarean'].map({'yes': 1, 'no': 0})
            x=data.iloc[:,0:5]
            y=data.iloc[:,-1]
            #y=y.dropna()
            #x = pd.DataFrame(row,columns=['Age', 'Deliveryno', 'bloodpressure', 'heartproblem', 'babyposition'])
            #print(x)
            #print(y)
            #encode the independent variable 'bp'
            labelencoder_x=LabelEncoder()
            x.iloc[:,2]=labelencoder_x.fit_transform(x.iloc[:,2])
            x.iloc[:,3]=labelencoder_x.fit_transform(x.iloc[:,3])
            x.iloc[:,4]=labelencoder_x.fit_transform(x.iloc[:,4])
            #print(x)
            #res = pd.DataFrame(row,columns=['Age', 'Deliveryno', 'bloodpressure', 'heartproblem', 'babyposition'])
            #print(res)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # 80% training and 20% test
            #print(x_test)
            #print(x_test)
            #rows=np.asarray(row)
            rows = pd.Series(row)
            print(rows)
            #print(type(rows))
            df = pd.DataFrame(rows)
            df = df.T
            df.columns = x_test.columns
            print(df)
            print("=================")
            print(x_test)
            
            clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
            #print(clf)
            # Train Decision Tree Classifer
            clf = clf.fit(x_train,y_train)
            #print(clf)
            decision_tree_pkl_filename = 'decision_tree_classifier_20170212.pkl'
            # Open the file to save as pkl file
            decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
            pickle.dump(clf, decision_tree_model_pkl)
            # Close the pickle instances
            decision_tree_model_pkl.close()
            decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
            decision_tree_model = pickle.load(decision_tree_model_pkl)
            print(decision_tree_model)
            filename = 'finalized_model.sav'

            pickle.dump(clf, open(filename, 'wb'))

            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.score(x_test, y_test)
            print(result)

            y_res = loaded_model.predict(df)
            print(y_res)
            #print(type(dno[1]))
            delno = int(dno[1])
            print(type(delno))
            if (y_res == [1]):
                flash("You may undergo Ceasarean delivery")
                return redirect(url_for('output'))

            elif((delno > 1 and typeofdeli == 'Caesarean')):
                flash("You may undergo Ceasarean delivery")    
                return redirect(url_for('output'))
            elif(y_res == [0] and (dno[1]=='1' and (typeofdeli == 'Normal' or typeofdeli=='--'))):
                flash("You may undergo Normal Delivery")
                return redirect(url_for('output'))
            elif(dno[1]=='1' and (typeofdeli == 'Normal' or typeofdeli=='Caesarean')):
                flash("Please select '--' in Type of delivery")
                return redirect(url_for('userInputs'))    

            #Predict the response for test dataset
            #y_pred = clf.predict(x_test)
            #print(y_pred)
            #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            #data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            #print(data)  
            #y_pred=y_pred.replace((1, 0), ('yes', 'no'), inplace=True) 
            #print(y_pred)
            # if (y_pred == 0 ):
            #     res = 'no'
            # else:
            #     res= 'yes'

            #print(res)        


            # feature_cols = ['Age', 'Deliveryno', 'bloodpressure', 'heartproblem', 'babyposition']
            

            # dot_data = tree.export_graphviz(clf, out_file=None,
            #                                feature_names=feature_cols,
            #                                class_names=['0','1'])
            # graph=pydotplus.graph_from_dot_data(dot_data)
            # img=Image(graph.create_png())
            #output_img = display(img.show())
            #print(img)


        #return redirect(url_for('userInputs'))
    return render_template('userinputs.html')

@app.route('/JSONdata')
def JSONdata():
    users = session.query(newUsers).all()
    for user in users:
        print(user.email)
        print(user.password)
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = "super secret key"
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
