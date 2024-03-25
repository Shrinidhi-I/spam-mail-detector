from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load the trained model and feature extractor
model = pickle.load(open("spam_ham_classifier.pkl", "rb"))
feature_extraction = pickle.load(open("feature_extractor.pkl", "rb"))

@app.route('/')
def home():
    # pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'spam_ham.jpg')
    return render_template('index.html')

@app.route('/display', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        text_of_email = request.form["text_of_email"]
        input_mail = [text_of_email]
        input_data_features = feature_extraction.transform(input_mail)
        prediction = model.predict(input_data_features)
        if prediction[0] == 1:
            result = "HAM(NOT SPAM MAIL)"
        else:
            result = "SPAM MAIL!!!"
        # pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'spam_ham.jpg')
        return render_template('result.html', result=result)

    # Handle GET method or initial rendering
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

