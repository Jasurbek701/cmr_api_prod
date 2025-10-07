from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import io
import tempfile
import os
from docx import Document
from docx2pdf import convert as docx2pdf

app = Flask(__name__)

MODEL_PATH = "best.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
CLASSES = ["cmr", "noncmr"]


def preprocess(img, img_size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    arr = img.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def convert_to_jpg(file, filename):
    """Converts any uploaded file to a JPG image"""
    ext = filename.split(".")[-1].lower()


    if ext == "pdf":
        pages = convert_from_bytes(file.read())
        img = pages[0].convert("RGB")
        return img


    elif ext in ["doc", "docx"]:
        with tempfile.TemporaryDirectory() as tmpdir:
 
            temp_doc = os.path.join(tmpdir, "temp.docx")
            file.save(temp_doc)

       
            pdf_path = os.path.join(tmpdir, "temp.pdf")
            docx2pdf(temp_doc, pdf_path)

   
            pages = convert_from_bytes(open(pdf_path, "rb").read())
            img = pages[0].convert("RGB")
            return img


    else:
        img = Image.open(file.stream).convert("RGB")
        return img


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "CMR API is running. Use POST /predict with an image, PDF, or Word document."})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    try:
 
        img = convert_to_jpg(file, filename)


        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        img_array = np.frombuffer(img_bytes.read(), np.uint8)
        cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)


        input_tensor = preprocess(cv_img)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})


        probs = outputs[0][0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return jsonify({
            "is_cmr": (CLASSES[pred_class] == "cmr"),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
