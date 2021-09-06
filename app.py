import joblib
import pandas as pb
import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictor', methods=['POST'])
def predictor():
    if request.method == 'POST':
        slen = request.form['slen']
        swid = request.form['swid']
        plen = request.form['plen']
        pwid = request.form['pwid']
        input_data = [slen, swid, plen, pwid]

        #chuyen doi thanh so
        chuyendoi = [float(i) for i in input_data]

        fil_data = np.array(chuyendoi).reshape(1, -1)

        dtmodel = joblib.load('data/dt_model.pkl')
        prediction = dtmodel.predict(fil_data)

        return render_template('index.html',slen=slen, swid=swid, plen=plen, pwid=pwid, prediction=prediction )


if __name__ == '__main__':
    app.run(debug=True)