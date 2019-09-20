from flask import Flask, flash, request, redirect, render_template
from flask_restful import Api
from flask_httpauth import HTTPBasicAuth
import os
from werkzeug.utils import secure_filename
from denoising import Denoising

path = os.path.dirname( os.path.realpath(__file__) )
noisy_speech_folder = os.path.join(path, 'datasets/noisy_speech')
sampled_noisy_speech_folder = os.path.join(path, 'datasets/sampled_noisy_speech')
modfolder = os.path.join(path, 'models')
denoise = Denoising(noisy_speech_folder=noisy_speech_folder, sampled_noisy_speech_folder=sampled_noisy_speech_folder, modfolder=modfolder)
UPLOAD_FOLDER = noisy_speech_folder

app = Flask(__name__)
app.secret_key = "speech denoising"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
api = Api(app)
auth = HTTPBasicAuth()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav']



@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded')
        return render_template('denoised_speech.html')
    else:
        flash('Currently only wav file types are supported')
        return redirect(request.url)

@app.route('/denoised_speech', methods=['GET', 'POST'])
def speech_denoising():
    denoise.sampling()
    denoise.inference()
    return render_template('success.html')


if __name__ == "__main__":
    app.run(debug=True)
