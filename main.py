import os
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import cv2
from operation import *


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Main code goes here
def proocessImage(filename, operation):
    print(f"The filename is {filename} and the operation is {operation}")
    img = cv2.imread(f"uploads/{filename}")
    match operation:
        case "cgray":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"static/{filename}", gray)
        case "snappy":
            snappy = add_text_on_image(img, 'Friday')
            cv2.imwrite(f"static/{filename}", snappy)
        case "histeql":
            snappy = histogram_equalization(img)
            cv2.imwrite(f"static/{filename}", snappy)
        case "remback":
            # snappy = bg_removal(img)
            snappy = remove_background(img)
            cv2.imwrite(f"static/{filename}", snappy)
        case "srkback":
            snappy = foreground(img, 1)
            cv2.imwrite(f"static/{filename}", snappy)
        case "yashback":
            snappy = foreground(img, 2)
            cv2.imwrite(f"static/{filename}", snappy)
        case "contour":
            snappy = add_counters(img)
            cv2.imwrite(f"static/{filename}", snappy)
        case "insta-summer":
            snappy = apply_filter(img, 'summer')
            cv2.imwrite(f"static/{filename}", snappy)
        case "insta-winter":
            snappy = apply_filter(img, 'winter')
            cv2.imwrite(f"static/{filename}", snappy)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        operation = request.form.get('operation')
        # return "POST request successful"
        if 'file' not in request.files:
            flash('No file part')
            return "ERROR: No file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "ERROR: No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            proocessImage(filename, operation)
            flash(
                f"Your image has been processed and is available <a href='/static/{filename}' target='_blank'> here. </a>")
            return render_template("index.html")


@app.route('/About')
def about():
    return render_template('about.html')


app.run(debug=True, port=5002)
