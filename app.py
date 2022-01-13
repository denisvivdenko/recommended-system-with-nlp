from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from recommendation_system import RecommendationSystem
from typing import List

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("data.csv")
users_list = list(df["crm_customer_id"].unique())

@app.route('/')
def home():
    return render_template('/index.html', users=users_list)

@app.route('/ra/connect', methods=['GET', 'POST'])
def connect_management():
    user = request.form.get('selected_class')
    return render_template('/results.html', result=str(model.predict(int(user))))
