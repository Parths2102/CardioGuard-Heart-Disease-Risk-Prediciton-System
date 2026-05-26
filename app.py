from pathlib import Path
import sqlite3
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "secret-key"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"

model_xgboost = joblib.load(BASE_DIR / "models/xgboost_heart_risk.pkl")
model_ai = model_xgboost 

# ================== Insights Logic =================
def get_health_insights(form):
    factors = []
    recommendations = []
    
    age = float(form.get("age_years", 0))
    bmi = float(form.get("bmi", 0))
    sys_bp = float(form.get("systolic_bp_mmHg", 0))
    dia_bp = float(form.get("diastolic_bp_mmHg", 0))
    hr = float(form.get("heart_rate_bpm", 0))
    activity = float(form.get("physical_activity_min_per_week", 0))
    stress = float(form.get("stress_level_1_10", 0))
    sleep = float(form.get("sleep_quality_score_1_10", 0))
    smoking = float(form.get("smoking_cigs_per_week", 0))
    alcohol = float(form.get("alcohol_ml_per_week", 0))

    # BP check
    if sys_bp > 140 or dia_bp > 90:
        factors.append("High Blood Pressure")
        recommendations.append("Reduce salt intake and monitor your blood pressure regularly.")
    elif sys_bp > 130 or dia_bp > 80:
        factors.append("Elevated Blood Pressure")
        recommendations.append("A balanced diet and regular exercise can help stabilize your BP.")

    # BMI check
    if bmi >= 30:
        factors.append("Obesity")
        recommendations.append("Consult a specialist for a personalized weight management program.")
    elif bmi >= 25:
        factors.append("Overweight (High BMI)")
        recommendations.append("Incorporate more fiber and whole grains into your daily meals.")

    # HR check
    if hr > 100:
        factors.append("High Resting Heart Rate")
        recommendations.append("Try breathing exercises to reduce stress and lower your resting heart rate.")
    elif hr < 60 and age < 60:
        factors.append("Low Resting Heart Rate")
        recommendations.append("While common in athletes, ensure you are not experiencing dizziness.")

    # Lifestyle
    if activity < 150:
        factors.append("Sedentary Lifestyle")
        recommendations.append("Try to achieve 150 minutes of moderate activity per week.")
    
    if smoking > 0:
        factors.append("Smoking Habit")
        recommendations.append("Smoking is a major heart risk factor. Seek support to quit or reduce intake.")

    if alcohol > 100:
        factors.append("Alcohol Consumption")
        recommendations.append("Excessive alcohol consumption is a heart risk factor. Consider reducing or eliminating intake.")

    if stress > 7:
        factors.append("High Stress Levels")
        recommendations.append("Practice mindfulness or yoga to manage daily stress effectively.")

    if sleep < 6:
        factors.append("Insufficient Sleep")
        recommendations.append("Prioritize 7-9 hours of restful sleep to allow your heart to recover.")

    # Fallback
    if not factors:
        factors.append("No major risk factors identified")
        recommendations.append("Keep up the healthy lifestyle! Regular checkups are still recommended.")

    return factors[:5], recommendations[:5]

# ================== database =====================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        age INTEGER,
        bmi REAL,
        blood_pressure REAL,
        diastolic_bp REAL,
        heart_rate REAL,
        result TEXT,
        time TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        full_name TEXT,
        gender TEXT,
        birth_date TEXT,
        blood_group TEXT,
        phone TEXT,
        address TEXT,
        profile_completed INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =================== prepare data ===================

def prepare(form):
    return np.array([[
        float(form.get("age_years", 0)),
        float(form.get("bmi", 0)),
        float(form.get("systolic_bp_mmHg", 0)),
        float(form.get("diastolic_bp_mmHg", 0)),
        float(form.get("heart_rate_bpm", 0)),
        float(form.get("previous_hypertensive_episodes", 0)),
        float(form.get("comorbidity_count", 0)),
        float(form.get("sleep_quality_score_1_10", 0)),
        float(form.get("physical_activity_min_per_week", 0)),
        float(form.get("stress_level_1_10", 0)),
        float(form.get("smoking_cigs_per_week", 0)),
        float(form.get("alcohol_ml_per_week", 0)),
    ]])

# ================= routes ====================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        flash("Please log in to access the heart health assessment.", "warning")
        return redirect(url_for("login"))
    return render_template("predict.html")

# ================= RESULT ====================

@app.route("/result", methods=["POST"])
def result():
    if "user" not in session:
        flash("Please log in to access the heart health assessment.", "warning")
        return redirect(url_for("login"))

    form = request.form
    data = prepare(form)

    # Internal check using the smart system
    pred = model_ai.predict(data)[0]
    prob = model_ai.predict_proba(data)[0][pred]
    risk_score = round(prob * 100, 2)

    if pred == 0:
        label = "LOW"
        color = "success"
    elif pred == 1:
        label = "MEDIUM"
        color = "warning"
    else:
        label = "HIGH"
        color = "danger"

    factors, recs = get_health_insights(form)

    # Prepare chart data
    chart_data = {
        "labels": ["BMI", "Systolic BP", "Diastolic BP", "Heart Rate"],
        "values": [
            float(form.get("bmi", 0)),
            float(form.get("systolic_bp_mmHg", 0)),
            float(form.get("diastolic_bp_mmHg", 0)),
            float(form.get("heart_rate_bpm", 0))
        ],
        "thresholds": [25, 120, 80, 100]  # Standard healthy references
    }

    # ===== Save to DB =====
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO predictions (username, age, bmi, blood_pressure, diastolic_bp, heart_rate, result, time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.get("user", "guest"),
        form.get("age_years"),
        form.get("bmi"),
        form.get("systolic_bp_mmHg"),
        form.get("diastolic_bp_mmHg"),
        form.get("heart_rate_bpm"),
        f"{label} Risk ({risk_score:.2f}%)",
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        risk_label=label,
        risk_score=risk_score,
        risk_color=color,
        factors=factors,
        recommendations=recs,
        chart_data=chart_data
    )

# ================= LOGIN ====================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()

        if user:
            session["user"] = username
            # Check if profile is completed
            c.execute("SELECT profile_completed FROM users WHERE username=?", (username,))
            profile_status = c.fetchone()[0]
            conn.close()
            
            flash("Login successful")
            if not profile_status:
                return redirect(url_for("profile_builder"))
            return redirect(url_for("home"))
        else:
            conn.close()
            flash("Invalid username or password")

    return render_template("login.html")

# ================= REGISTER ====================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()

            flash("Registered successfully")
            return redirect(url_for("login"))

        except Exception as e:
            flash("Username already exists")

    return render_template("register.html")

# ================= LOGOUT ====================

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out")
    return redirect(url_for("login"))

# ================= OTHER ====================

@app.route("/profile-builder", methods=["GET", "POST"])
def profile_builder():
    if "user" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        full_name = request.form.get("full_name")
        gender = request.form.get("gender")
        birth_date = request.form.get("birth_date")
        blood_group = request.form.get("blood_group")
        phone = request.form.get("phone")
        address = request.form.get("address")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE users 
            SET full_name=?, gender=?, birth_date=?, blood_group=?, phone=?, address=?, profile_completed=1 
            WHERE username=?
        """, (full_name, gender, birth_date, blood_group, phone, address, session["user"]))
        conn.commit()
        conn.close()
        
        flash("Profile built successfully!")
        return redirect(url_for("profile"))

    return render_template("profile_builder.html")

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (session["user"],))
    user_data = c.fetchone()

    c.execute("SELECT * FROM predictions WHERE username=? ORDER BY id DESC", (session["user"],))
    assessments = c.fetchall()
    conn.close()
    
    if not user_data["profile_completed"]:
        return redirect(url_for("profile_builder"))
        
    return render_template("profile.html", user=user_data, assessments=assessments)

@app.route("/delete-assessment/<int:assessment_id>", methods=["POST"])
def delete_assessment(assessment_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id=? AND username=?", (assessment_id, session["user"]))
    conn.commit()
    conn.close()
    
    flash("Assessment deleted successfully", "success")
    return redirect(url_for("profile"))

@app.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if request.method == "POST":
        full_name = request.form.get("full_name")
        gender = request.form.get("gender")
        birth_date = request.form.get("birth_date")
        blood_group = request.form.get("blood_group")
        phone = request.form.get("phone")
        address = request.form.get("address")

        c.execute("""
            UPDATE users 
            SET full_name=?, gender=?, birth_date=?, blood_group=?, phone=?, address=? 
            WHERE username=?
        """, (full_name, gender, birth_date, blood_group, phone, address, session["user"]))
        conn.commit()
        conn.close()
        
        flash("Profile updated successfully")
        return redirect(url_for("profile"))

    # GET: Fetch current data to pre-fill the form
    c.execute("SELECT * FROM users WHERE username=?", (session["user"],))
    user_data = c.fetchone()
    conn.close()
    
    return render_template("edit_profile.html", user=user_data)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ================= RUN ====================

if __name__ == "__main__":
    app.run(debug=True)