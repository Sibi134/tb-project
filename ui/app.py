import gradio as gr
import pandas as pd
import plotly.express as px
import sys
import os

# ------------------------------------------------
# Resolve project root
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from prediction import predict_tb
from resource_planner import allocate_resources
from ai_assistant import tb_chatbot, explain_prediction

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
data_path = os.path.join(BASE_DIR, "cleaned_data", "tb_cleaned.csv")
data = pd.read_csv(data_path)

beds_df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_data", "beds_cleaned.csv"))
doctors_df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_data", "doctors_cleaned.csv"))

data = data.merge(beds_df, on="Country", how="left")
data = data.merge(doctors_df, on="Country", how="left")

data["Available_Beds"] = (data["Beds_per_1000"] / 1000) * data["Population"]
data["Available_Doctors"] = (data["Doctors_per_1000"] / 1000) * data["Population"]

data["Available_Beds"] = data["Available_Beds"].fillna(data["Population"] * 0.0015)
data["Available_Doctors"] = data["Available_Doctors"].fillna(data["Population"] * 0.0009)

data["TB_Rate"] = (data["TB_Cases"] / data["Population"]) * 100000

# ------------------------------------------------
# Intelligence Functions
# ------------------------------------------------
def generate_insights(predicted_cases, beds_needed, doctors_needed):
    insights = []

    if predicted_cases > 1_000_000:
        insights.append("⚠️ High TB burden detected.")
    elif predicted_cases > 500_000:
        insights.append("⚠️ Moderate TB burden.")
    else:
        insights.append("✅ TB under control.")

    if beds_needed > predicted_cases * 0.2:
        insights.append("🚨 High hospitalization demand.")

    if doctors_needed > predicted_cases * 0.05:
        insights.append("⚠️ Doctor load increasing.")

    return "\n".join(insights)


def get_risk_level(predicted_cases):
    if predicted_cases > 1_000_000:
        return "🔴 High Risk"
    elif predicted_cases > 500_000:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"


def resource_score(predicted_cases, beds_needed, doctors_needed, kits):
    score = ((beds_needed + doctors_needed + kits) / (predicted_cases * 3)) * 100
    return round(score, 2)

# ------------------------------------------------
# TRIAGE SYSTEM
# ------------------------------------------------
def patient_triage(predicted_cases):
    return int(predicted_cases * 0.2), int(predicted_cases * 0.5), int(predicted_cases * 0.3)

# ------------------------------------------------
# ALLOCATION SYSTEM
# ------------------------------------------------
def hospital_allocation(critical, moderate, mild, available_beds):
    return {
        "critical": min(critical, int(available_beds * 0.5)),
        "moderate": min(moderate, int(available_beds * 0.3)),
        "mild": min(mild, int(available_beds * 0.2)),
    }

# ------------------------------------------------
# 🆕 PATIENT ALLOCATION
# ------------------------------------------------
def allocate_patient(risk):
    if "High" in risk:
        return "🏥 Immediate Hospital Admission (Isolation Ward Required)"
    elif "Medium" in risk:
        return "🩺 OPD Consultation + Diagnostic Tests Recommended"
    else:
        return "🏠 Home Care + Monitoring"

# ------------------------------------------------
# PERSONAL SIMPLE DETECTION (UPDATED)
# ------------------------------------------------
def personal_tb(age, cough, fever, weight_loss, chest_pain):
    score = 0
    if cough: score += 2
    if fever: score += 2
    if weight_loss: score += 3
    if chest_pain: score += 2
    if age > 50: score += 1

    if score >= 7:
        diagnosis = "🔴 TB Positive"
        risk = "High Risk"
    elif score >= 4:
        diagnosis = "🟠 TB Positive"
        risk = "Medium Risk"
    else:
        diagnosis = "🟢 TB Negative"
        risk = "Low Risk"

    allocation = allocate_patient(risk)
    bar = "█" * score + "░" * (10 - score)

    return f"""
### 🧑 Patient Analysis

Diagnosis: {diagnosis}  
Risk Level: {risk}  
Score: {score}/10  

### 📊 Severity Meter  
{bar}

### 🏥 Recommended Action  
{allocation}
"""

# ------------------------------------------------
# PERSONAL INSIGHTS
# ------------------------------------------------
def personal_insights(age, cough, fever, weight_loss, chest_pain):
    insights = []

    symptom_count = sum([cough, fever, weight_loss, chest_pain])

    if symptom_count >= 3:
        insights.append("⚠️ Multiple symptoms detected — high likelihood of TB.")
    elif symptom_count == 2:
        insights.append("⚠️ Moderate symptoms — medical checkup recommended.")
    else:
        insights.append("✅ Few symptoms — lower immediate risk.")

    if cough:
        insights.append("🫁 Persistent cough is a major TB indicator.")
    if weight_loss:
        insights.append("⚖️ Weight loss may indicate disease progression.")
    if fever:
        insights.append("🌡️ Fever suggests infection response.")
    if chest_pain:
        insights.append("💢 Chest pain may indicate lung involvement.")
    if age > 50:
        insights.append("👴 Age increases vulnerability.")

    insights.append("📌 Recommendation: Consult a doctor and take a TB test.")

    return "### 💡 AI Insights\n\n" + "\n".join(insights)

# ------------------------------------------------
# VISUALIZATION FUNCTIONS
# ------------------------------------------------
def show_trend(country):
    df = data[data["Country"] == country].reset_index(drop=True)

    fig = px.line(
        df,
        x="Year",
        y="TB_Cases",
        title=f"Tuberculosis Trend in {country}",
        labels={"Year": "Year", "TB_Cases": "TB Cases"},
        markers=True
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show_tb_map():
    df = data.groupby("Country").agg({
        "TB_Cases": "sum",
        "Population": "mean"
    }).reset_index()

    df["TB_Rate"] = (df["TB_Cases"] / df["Population"]) * 100000

    return px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="TB_Rate",
        title="Global TB Incidence Rate",
        color_continuous_scale="Reds"
    )


def compare_countries(selected_countries):
    if not selected_countries:
        return None

    df = data[data["Country"].isin(selected_countries)].reset_index(drop=True)

    return px.line(
        df,
        x="Year",
        y="TB_Cases",
        color="Country",
        title="TB Cases Comparison by Country",
        markers=True
    )

# ------------------------------------------------
# MAIN SYSTEM (FULL VERSION RESTORED)
# ------------------------------------------------
def tb_system(country, year, population):

    predicted_cases = predict_tb(year, population)
    beds_req, doctors_req, kits = allocate_resources(predicted_cases)

    country_data = data[data["Country"] == country].iloc[0]

    available_beds = int(country_data["Available_Beds"] * 0.3)
    available_doctors = int(country_data["Available_Doctors"] * 0.4)

    beds_needed = int(predicted_cases * 0.2)
    doctors_needed = int(predicted_cases * 0.05)

    bed_gap = beds_needed - available_beds
    doctor_gap = doctors_needed - available_doctors

    critical, moderate, mild = patient_triage(predicted_cases)
    allocation = hospital_allocation(critical, moderate, mild, available_beds)

    if bed_gap > available_beds * 0.5:
        alert = "🚨 CRITICAL: System overwhelmed"
    elif bed_gap > 0:
        alert = "⚠️ Moderate strain"
    else:
        alert = "✅ Stable"

    risk = get_risk_level(predicted_cases)
    score = resource_score(predicted_cases, beds_needed, doctors_needed, kits)
    insights = generate_insights(predicted_cases, beds_needed, doctors_needed)

    return f"""
### Country: {country}

### Predicted TB Cases
{predicted_cases:,}

### Risk Level
{risk}

### 🚨 System Status
{alert}

---

Beds Needed: {beds_needed:,}  
Beds Available: {available_beds:,}

Doctors Needed: {doctors_needed:,}  
Doctors Available: {available_doctors:,}

---

Critical: {critical:,}  
Moderate: {moderate:,}  
Mild: {mild:,}

---

TB Test Kits Required: {kits:,}

Score: {score}%

Insights:
{insights}
"""

# ------------------------------------------------
# Dashboard Stats
# ------------------------------------------------
countries = sorted(data["Country"].unique())
total_cases = int(data["TB_Cases"].sum())
total_countries = data["Country"].nunique()
avg_cases = int(data["TB_Cases"].mean())

# ------------------------------------------------
# UI (FULLY RESTORED)
# ------------------------------------------------
css = """
body{background-color:#0f172a;}
.main-title{text-align:center;font-size:36px;font-weight:700;color:white;}
.subtitle{text-align:center;color:#cbd5f5;font-size:18px;margin-bottom:20px;}
.card{background:#1e293b;padding:20px;border-radius:10px;text-align:center;box-shadow:0px 4px 10px rgba(0,0,0,0.4);}
.card-title{font-size:15px;color:#94a3b8;}
.card-value{font-size:26px;font-weight:bold;color:white;}
button{border-radius:8px !important;}
"""

# ✅ FIXED (removed title ONLY)
with gr.Blocks(css=css) as demo:

    gr.HTML("""
    <div class='main-title'>AI-Driven Tuberculosis Monitoring System</div>
    <div class='subtitle'>
    Public Health Analytics Dashboard for TB Prediction and Resource Planning
    </div>
    """)

    with gr.Row():
        for title, value in [
            ("Total TB Cases", total_cases),
            ("Countries Covered", total_countries),
            ("Average Cases", avg_cases)
        ]:
            with gr.Column():
                gr.HTML(f"""
                <div class='card'>
                <div class='card-title'>{title}</div>
                <div class='card-value'>{value}</div>
                </div>
                """)

    with gr.Tabs():

        with gr.Tab("Disease Monitoring"):
            country_input = gr.Dropdown(countries, label="Select Country")
            trend_output = gr.Plot()
            gr.Button("Generate Trend").click(show_trend, country_input, trend_output)

        with gr.Tab("TB Hotspot Map"):
            map_output = gr.Plot()
            gr.Button("Generate Map").click(show_tb_map, outputs=map_output)

        with gr.Tab("Country Comparison Dashboard"):
            selector = gr.Dropdown(countries, multiselect=True, label="Select Countries")
            comp_output = gr.Plot()
            gr.Button("Compare").click(compare_countries, selector, comp_output)

        with gr.Tab("Prediction & Resource Planning"):
            c = gr.Dropdown(countries, label="Select Country")
            y = gr.Number(value=2027, label="Year")
            p = gr.Number(value=1400000000, label="Population")

            out = gr.Markdown()
            ai_out = gr.Markdown()

            gr.Button("Predict").click(tb_system, [c, y, p], out)
            gr.Button("Generate AI Insight").click(explain_prediction, out, ai_out)

        with gr.Tab("AI Healthcare Assistant"):
            q = gr.Textbox(label="Ask a question")
            out = gr.Markdown()
            gr.Button("Ask AI").click(tb_chatbot, q, out)

        with gr.Tab("Personal TB Detection"):
            age = gr.Number(label="Age", value=30)
            cough = gr.Checkbox(label="Persistent Cough")
            fever = gr.Checkbox(label="Fever")
            weight_loss = gr.Checkbox(label="Weight Loss")
            chest_pain = gr.Checkbox(label="Chest Pain")

            personal_output = gr.Markdown()
            insight_output = gr.Markdown()

            gr.Button("Analyze Patient").click(
                personal_tb,
                inputs=[age, cough, fever, weight_loss, chest_pain],
                outputs=personal_output
            )

            gr.Button("Insights").click(
                personal_insights,
                inputs=[age, cough, fever, weight_loss, chest_pain],
                outputs=insight_output
            )

# ✅ FINAL FIX (Render)
port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)