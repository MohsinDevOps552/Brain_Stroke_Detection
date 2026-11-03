from flask import Flask, render_template, request, send_from_directory, send_file
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from fpdf import FPDF
import datetime

app = Flask(__name__, static_url_path='', static_folder='static')
model = load_model('brain.h5')

UPLOAD_FOLDER = 'static/uploads'
PDF_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ---------- Image Validation ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Check if it's a valid image
            if img.format not in ['JPEG', 'JPG', 'PNG']:
                return False
        return True
    except:
        return False

# ---------- Prediction ----------
def predict_tumor(image_path):
    try:
        if not is_valid_image(image_path):
            return "Invalid image format. Please upload a valid MRI image (JPG, JPEG, PNG).", 0.0

        img = Image.open(image_path).convert('RGB').resize((128, 128))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions)

        if confidence < 0.5:
            return "Low confidence: Unable to determine tumor type.", confidence

        index = np.argmax(predictions)
        result = class_labels[index]
        return ("No Tumor" if result == "notumor" else f"Tumor: {result}"), confidence
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

# ---------- PDF Report Generator ----------
def generate_pdf(patient_info, result, confidence, image_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Brain Tumor Detection Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    for label, value in patient_info.items():
        pdf.cell(0, 10, f"{label}: {value}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, f"Diagnosis Result: {result}", ln=True)
    pdf.cell(0, 10, f"Confidence Score: {confidence:.2%}", ln=True)

    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    if os.path.exists(image_path):
        pdf.ln(10)
        pdf.image(image_path, x=10, w=100)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f"report_{timestamp}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    pdf.output(pdf_path)
    return pdf_filename

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        symptoms = request.form.get('symptoms')
        doctor = request.form.get('doctor')

        if not file or file.filename == '':
            return render_template('index.html', error="No file selected.")

        if not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file type. Only JPG, JPEG, and PNG are allowed.")

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result, confidence = predict_tumor(filepath)

        # In case of error during prediction
        if result.startswith("Invalid") or result.startswith("Error") or result.startswith("Low confidence"):
            return render_template('index.html', error=result)

        patient_info = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Doctor": doctor
        }

        try:
            pdf_filename = generate_pdf(patient_info, result, confidence, filename)
            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f'/uploads/{filename}',
                report_path=f'/download-report/{pdf_filename}'
            )
        except Exception as e:
            return render_template(
                'index.html',
                error=f"Failed to generate report: {str(e)}",
                result=result,
                file_path=f'/uploads/{filename}'
            )

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/download-report/<filename>')
def download_report(filename):
    return send_file(
        os.path.join(PDF_FOLDER, filename),
        as_attachment=True,
        download_name=f"Brain_Report_{filename}"
    )

# ---------- Run App ----------
if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
    app.run(debug=True)
