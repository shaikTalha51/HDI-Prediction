"""
app.py
------
Epic 8: Building the Flask Web Application

Routes:
    /            Home page - project introduction
    /predict     GET: input form | POST: run prediction, show result
    /insights    EDA charts generated during training
"""

import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------
with open("hdi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

FEATURES = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
]


def classify_hdi(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    return "Low"


TIER_INFO = {
    "Very High": {
        "color": "#1f9d6b",
        "blurb": "This country is among the most developed nations, with strong "
                  "outcomes across health, education, and income.",
    },
    "High": {
        "color": "#3d8bd4",
        "blurb": "This country shows solid development outcomes, with room to "
                  "grow toward the very-high tier.",
    },
    "Medium": {
        "color": "#e0a82e",
        "blurb": "This country shows moderate development. Targeted investment "
                  "in healthcare, education, or income generation could meaningfully "
                  "raise outcomes.",
    },
    "Low": {
        "color": "#d4533d",
        "blurb": "This country faces significant developmental challenges. "
                  "Priority intervention in health, education, and income is recommended.",
    },
}


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        values = [float(request.form[f]) for f in FEATURES]
    except (KeyError, ValueError):
        return render_template(
            
            "predict.html",
            error="Please fill in all fields with valid numbers.",
        )

    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = float(model.predict(X_scaled)[0])
    pred = max(0.0, min(1.0, pred))
    tier = classify_hdi(pred)

    return render_template(
        "result.html",
        score=round(pred, 3),
        tier=tier,
        tier_color=TIER_INFO[tier]["color"],
        tier_blurb=TIER_INFO[tier]["blurb"],
        inputs=dict(zip(FEATURES, values)),
    )


@app.route("/insights")
def insights():
    return render_template("insights.html", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
