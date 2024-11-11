from flask import Flask, render_template, request, jsonify
import os
from model import load_model, predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model()  # Load the model once when the app starts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save the uploaded file
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Run prediction
            predicted_class, accuracy = predict_image(file_path, model)
            
            # Return prediction result as JSON
            return jsonify({
                "predicted_class": predicted_class,
                "accuracy": accuracy
            })
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
