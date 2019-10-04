from flask import Flask, flash, request, redirect, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from denoising import Denoising

path = os.path.dirname( os.path.realpath(__file__) )
noisy_speech_folder = os.path.join(path, 'datasets/noisy_speech')
sampled_noisy_speech_folder = os.path.join(path, 'datasets/sampled_noisy_speech')
modfolder = os.path.join(path, 'models')
denoise = Denoising(noisy_speech_folder=noisy_speech_folder, sampled_noisy_speech_folder=sampled_noisy_speech_folder, modfolder=modfolder)
UPLOAD_FOLDER = noisy_speech_folder
DOWNLOAD_FOLDER = os.path.join(path,'datasets/sampled_noisy_speech_denoised')

app = Flask(__name__)
app.secret_key = "speech denoising"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav']



@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if allowed_file(file.filename):
            filename = file.filename
            filename = secure_filename(filename)
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                flash('File already exists. Please choose a different file')
                return render_template('upload.html')
            else:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File successfully uploaded')
                return render_template('denoised_speech.html')
        else:
            flash('Only wav files are supported currently')
            return render_template('upload.html')
    else:
        return redirect(request.url)

@app.route('/denoised_speech', methods=['GET', 'POST'])
def speech_denoising():
    denoise.sampling()
    denoise.inference()
    return render_template('success.html')

@app.route('/download', methods=['GET'])
def download_file():
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
