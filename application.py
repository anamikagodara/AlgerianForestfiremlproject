import pickle
from flask import Flask, request, render_template
import numpy as np

# Create Flask App
application = Flask(__name__)
app = application

# Load Model and Scaler
ridge_model = pickle.load(open("model/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Prediction Route
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "POST":

        # Get form values
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Convert into array
        data = [[
            Temperature,
            RH,
            WS,
            Rain,
            FFMC,
            DMC,
            ISI,
            Classes,
            Region
        ]]

        # Scale Data
        scaled_data = standard_scaler.transform(data)

        # Prediction
        prediction = ridge_model.predict(scaled_data)

        result = round(prediction[0], 2)

        # Return result to HTML
        return render_template("home.html", result=result)

    else:
        return render_template("home.html")

# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)