import os
from image_caption_generator import ImageCaptionGenerator
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = os.path.join('static', 'uploaded_images')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('get_caption',
                                    filename=filename))
    return render_template("index.html")

@app.route('/caption/<path:filename>', methods=['GET'])
def get_caption(filename):
    imgcptgen = ImageCaptionGenerator()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model, tokenizer, max_length = imgcptgen.testing_params()
    caption = imgcptgen.test(model, tokenizer, max_length, full_filename)
    return render_template("caption.html", captioned_image = 'uploaded_images/' + filename, caption = caption)
