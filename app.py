import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for better UI
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* Reduce top spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px;
}

/* Heartbeat Background Animation (Subtle ECG) */
@keyframes heartbeat-bg {
    0% { opacity: 0.15; }
    50% { opacity: 0.35; }   /* Clearer pulse */
    100% { opacity: 0.15; }
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    /* Soft Blue medical cross/heartbeat pattern - Professional & Subtle */
    background-image: url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" stroke="%233498db" stroke-width="1"%3e%3cpath d="M10 50 H20 L25 35 L35 65 L40 50 H50" /%3e%3c/svg%3e');
    background-size: 100px 100px;
    background-repeat: space;
    opacity: 0.15; /* Fallback opacity */
    z-index: 0;
    animation: heartbeat-bg 4s ease-in-out infinite;
    pointer-events: none;
}

/* Navbar container style */
/* Navbar container style */
div[data-testid="stRadio"] > div {
    display: flex;
    justify-content: center;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.95); /* White background */
    backdrop-filter: blur(5px);
    padding: 10px 20px;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    margin-top: 10px;
    border: 1px solid rgba(255, 255, 255, 0.5);
}

div[role="radiogroup"] label {
    margin-right: 15px;
    padding: 8px 16px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    background-color: #f8f9fa; /* Slight off-white background for pills */
    color: #445566;
    font-weight: 500;
    border-radius: 20px;
}

div[role="radiogroup"] label:hover {
    color: #2980b9;
    background-color: #e1f5fe;
    transform: translateY(-2px);
}

/* Selected item styling */
div[role="radiogroup"] label[data-checked="true"] {
    background-color: #3498db !important;
    color: white !important;
    border: none !important;
    border-radius: 20px;
    box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
}

div[role="radiogroup"] {
    flex-direction: row;
    display: flex;
    justify-content: center;
    flex-wrap: wrap; 
    gap: 5px;
}

/* Main content header */
.main-header {
    background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.main-header h1 {
    color: white !important;
    font-weight: 700;
}

.main-header h3 {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 400;
}

/* Cards */
.card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); /* Increased shadow for white-on-white */
    margin-bottom: 25px;
    border: 1px solid #f0f0f0; /* Subtle border */
    border-top: 5px solid #3498db;
    transition: transform 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.card h3 {
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Form styling */
[data-testid="stForm"] {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
}

/* Result boxes */
.result-low {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.result-high {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* Warnings and Info */
.info-box {
    background: #e1f5fe;
    border-left: 5px solid #039be5;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    color: #01579b;
}

.warning-box {
    background: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    color: #e65100;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Button Style */
.stButton button {
    background-color: #3498db;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    transition: all 0.3s;
}

.stButton button:hover {
    background-color: #2980b9;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load saved model objects
# -----------------------------
@st.cache_resource
def load_models():
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("models/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return model, scaler, feature_columns, mappings

model, scaler, feature_columns, mappings = load_models()
num_cols = ['height', 'weight', 'ap_hi', 'ap_lo', 'age_years']

# -----------------------------
# Global Auto-Scroll
# -----------------------------
st.markdown("""
    <script>
        var element = window.parent.document.querySelector('.main .block-container');
        if (element) {
            element.scrollTop = 0;
        }
    </script>
""", unsafe_allow_html=True)

# -----------------------------
# Global Header (Conditional - Home Only)
# -----------------------------
# Get current page from session state to decide on header rendering
current_page = st.session_state.get("navigation", "🏠 Home")

if current_page == "🏠 Home":
    st.markdown("""
<div class="main-header">
    <h1>CardioCare AI</h1>
    <h3>Your Intelligent Cardiovascular Health Assistant</h3>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Top Navigation Bar
# -----------------------------


# Horizontal Radio Button used as Navbar
page = st.radio(
    "Navigation",
    ["🏠 Home", "🔍 Predict Risk", "📊 Health Dashboard", "💡 Prevention Tips", 
     "📚 About Parameters", "📈 Model Analysis", "ℹ️ About Project"],
    horizontal=True,
    label_visibility="collapsed",
    key="navigation"
)

st.markdown("---")



# -----------------------------
# Helper Functions
# -----------------------------
def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_blood_pressure_category(systolic, diastolic):
    """Get BP category"""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"

# -----------------------------
# Page 1: Home Page
# -----------------------------
if page == "🏠 Home":
    # Header moved to global top
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", "73.5%", "±2%")
    with col2:
        st.metric("Parameters Analyzed", "12", "Health Factors")
    with col3:
        st.metric("Early Detection", "85%", "Success Rate")
    
    st.markdown("""
    <div class="card">
        <h3>Why Monitor Heart Health?</h3>
        <p>Cardiovascular diseases are the leading cause of death globally. 
        Early detection can prevent up to 80% of heart attacks and strokes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>How It Works</h3>
        <ol>
            <li>📝 Enter your health parameters in the Predict Risk page</li>
            <li>🤖 Our AI model analyzes 12+ risk factors</li>
            <li>📊 Get instant risk assessment with personalized insights</li>
            <li>💡 Receive actionable prevention tips</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📈 Global Heart Health Statistics")
    stats_data = {
        "Statistic": ["Annual Deaths", "Preventable Cases", "Early Detection Impact", "Lifestyle Improvement Benefit"],
        "Value": ["17.9 Million", "80%", "Reduces risk by 50%", "Improves outcomes by 60%"],
        "Impact": ["High", "High", "Medium", "High"]
    }
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <h3>Ready to Check Your Heart Health?</h3>
        <p>Navigate to <b>Predict Risk</b> in the sidebar to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Page 2: Predict Risk (Main Prediction Page)
# -----------------------------
# -----------------------------
# Page 2: Predict Risk (Main Prediction Page)
# -----------------------------
elif page == "🔍 Predict Risk":
    
    # Initialize session state for this page
    if 'prediction_state' not in st.session_state:
        st.session_state['prediction_state'] = 'input'
    
    # INPUT STATE
    if st.session_state['prediction_state'] == 'input':
        st.write("Enter your health parameters for AI-powered analysis")
        
        # Create form in columns
        with st.form("heart_form"):
            st.markdown("<div class='card'><h3>👤 Personal Information</h3></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                age_years = st.number_input("Age (years)", 18, 100, 45, help="Your current age")
                height = st.number_input("Height (cm)", 120, 220, 165)
                weight = st.number_input("Weight (kg)", 30, 200, 70)
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                smoke = st.selectbox("Do you smoke?", ["no", "yes"], help="Regular tobacco smoking")
                alco = st.selectbox("Alcohol consumption", ["no", "yes"], help="Regular alcohol intake")
                active = st.selectbox("Physical activity", ["no", "yes"], help="Regular exercise or active lifestyle")
            
            st.markdown("<div class='card'><h3>🩺 Medical Information</h3></div>", unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                ap_hi = st.number_input("Systolic BP (mmHg)", 80, 250, 120, help="Higher number in blood pressure reading")
                cholesterol = st.selectbox("Cholesterol Level", 
                                          ["normal", "above_normal", "well_above"],
                                          help="Your cholesterol level")
            
            with col4:
                ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 150, 80, help="Lower number in blood pressure reading")
                gluc = st.selectbox("Glucose Level",
                                   ["normal", "above_normal", "well_above"],
                                   help="Your blood glucose level")
            
            # Calculate real-time metrics for display (optional, can be removed if confusing in form)
            # Keeping it simple for the form view
            
            submit = st.form_submit_button("🚀 Predict My Risk", use_container_width=True)
        
        if submit:
            with st.spinner("Analyzing your health data with AI..."):
                # Create input dataframe
                df_input = pd.DataFrame(0, index=[0], columns=feature_columns)
                
                # Numeric values
                df_input.loc[0, "height"] = height
                df_input.loc[0, "weight"] = weight
                df_input.loc[0, "ap_hi"] = ap_hi
                df_input.loc[0, "ap_lo"] = ap_lo
                df_input.loc[0, "age_years"] = age_years
                
                # Scale numeric columns
                df_input[num_cols] = scaler.transform(df_input[num_cols])
                
                # Encoded categorical values
                df_input.loc[0, "cholesterol"] = mappings["cholesterol"][cholesterol]
                df_input.loc[0, "gluc"] = mappings["gluc"][gluc]
                df_input.loc[0, "smoke"] = mappings["smoke"][smoke]
                df_input.loc[0, "alco"] = mappings["alco"][alco]
                df_input.loc[0, "active"] = mappings["active"][active]
                
                # Gender one-hot
                gender_map = mappings["gender"][gender]
                df_input.loc[0, "gender_Male"] = gender_map["gender_Male"]
                df_input.loc[0, "gender_Female"] = gender_map["gender_Female"]
                
                # Prediction
                probability = model.predict_proba(df_input)[0][1]
                prediction = int(probability >= 0.5)
                
                # Calculated metrics for report
                bmi = calculate_bmi(height, weight)
                bmi_category = get_bmi_category(bmi)
                bp_category = get_blood_pressure_category(ap_hi, ap_lo)
                
                # Risk factors
                risk_score = 0
                if age_years > 50: risk_score += 1
                if bmi > 25: risk_score += 1
                if smoke == "yes": risk_score += 1
                
                medical_risk = 0
                if cholesterol != "normal": medical_risk += 1
                if gluc != "normal": medical_risk += 1
                if ap_hi > 140 or ap_lo > 90: medical_risk += 1
                
                total_risk = risk_score + medical_risk

                # Save to session state
                st.session_state['last_prediction'] = {
                    'probability': probability,
                    'prediction': prediction,
                    'age_years': age_years,
                    'gender': gender,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'bmi_category': bmi_category,
                    'ap_hi': ap_hi,
                    'ap_lo': ap_lo,
                    'bp_category': bp_category,
                    'smoke': smoke,
                    'alco': alco,
                    'active': active,
                    'cholesterol': cholesterol,
                    'gluc': gluc,
                    'lifestyle_risk': risk_score,
                    'medical_risk': medical_risk,
                    'total_risk': total_risk
                }
                
                # Change state and rerun
                st.session_state['prediction_state'] = 'result'
                st.rerun()

    # RESULT STATE
    elif st.session_state['prediction_state'] == 'result':
        # Retrieve data
        data = st.session_state.get('last_prediction', {})
        
        # Auto-scroll to top
        st.markdown("""
            <script>
                var element = window.parent.document.querySelector('.main .block-container');
                if (element) {
                    element.scrollTop = 0;
                }
            </script>
        """, unsafe_allow_html=True)
        
        st.markdown("### Here is your personalized AI health assessment")
        
        if data:
            prediction = data['prediction']
            probability = data['probability']
            
            if prediction == 1:
                st.markdown(
                    f"""<div class='result-high'>
                    ⚠️ <b>HIGH RISK DETECTED</b><br>
                    Risk Probability: <b>{probability:.1%}</b><br>
                    <small>Please consult a healthcare professional</small>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                st.markdown("""
<div class="warning-box">
    <h4>⚠️ Immediate Recommendations:</h4>
    <ul>
        <li>Consult a cardiologist or healthcare provider</li>
        <li>Schedule a comprehensive health checkup</li>
        <li>Monitor your blood pressure daily</li>
        <li>Consider lifestyle modifications</li>
        <li>Avoid smoking and limit alcohol</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div class='result-low'>
                    ✅ <b>LOW RISK</b><br>
                    Risk Probability: <b>{probability:.1%}</b><br>
                    <small>Continue maintaining healthy habits!</small>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                st.markdown("""
<div class="info-box">
    <h4>🎉 Great Going!</h4>
    <p>Your current health parameters indicate low cardiovascular risk. 
    Keep up the good work with regular exercise, balanced diet, and annual checkups.</p>
</div>
""", unsafe_allow_html=True)
            
            # Risk factors breakdown
            st.markdown("### 🔍 Risk Factors Analysis")
            col8, col9, col10 = st.columns(3)
            
            with col8:
                st.metric("Lifestyle Risk", data['lifestyle_risk'], "/3 factors")
            
            with col9:
                st.metric("Medical Risk", data['medical_risk'], "/3 factors")
            
            with col10:
                st.metric("Total Risk Factors", data['total_risk'], "/6 possible")
            
            # Metrics
            st.markdown("### 📊 Your Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("BMI", f"{data['bmi']:.1f}", data['bmi_category'])
            with col_m2:
                st.metric("Blood Pressure", f"{data['ap_hi']}/{data['ap_lo']}", data['bp_category'])
            with col_m3:
                st.metric("Age", data['age_years'], "Years")

            # Report Generation
            report = f"""
            HEART HEALTH REPORT
            ===================
            
            Personal Information:
            - Age: {data['age_years']} years
            - Gender: {data['gender']}
            - Height: {data['height']} cm
            - Weight: {data['weight']} kg
            - BMI: {data['bmi']:.1f} ({data['bmi_category']})
            
            Lifestyle Factors:
            - Smoking: {data['smoke']}
            - Alcohol: {data['alco']}
            - Physical Activity: {data['active']}
            
            Medical Parameters:
            - Blood Pressure: {data['ap_hi']}/{data['ap_lo']} mmHg ({data['bp_category']})
            - Cholesterol: {data['cholesterol']}
            - Glucose: {data['gluc']}
            
            PREDICTION RESULTS:
            - Risk Probability: {data['probability']:.1%}
            - Risk Level: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}
            - Total Risk Factors: {data['total_risk']}/6
            
            Recommendations:
            {'Consult a healthcare professional immediately' if prediction == 1 
             else 'Continue maintaining healthy lifestyle habits'}
            
            Report generated by CardioCare AI
            """
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="📥 Download Full Report",
                    data=report,
                    file_name="heart_health_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_btn2:
                if st.button("⬅️ Check Another Patient", use_container_width=True):
                    st.session_state['prediction_state'] = 'input'
                    st.rerun()

# -----------------------------
# Page 3: Health Dashboard (Simplified without Plotly)
# -----------------------------
elif page == "📊 Health Dashboard":
    
    st.markdown("### 📈 Risk Factor Distribution")
    
    # Sample data for charts
    risk_factors = ['High BP', 'Cholesterol', 'Smoking', 'Obesity', 'Diabetes', 'Inactivity']
    prevalence = [45, 38, 25, 32, 18, 40]
    
    # Create bar chart with matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(risk_factors, prevalence, color=['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5', '#ab47bc', '#26c6da'])
    ax.set_xlabel('Prevalence (%)')
    ax.set_title('Common Cardiovascular Risk Factors')
    ax.bar_label(bars, fmt='%d%%')
    st.pyplot(fig)
    
    st.markdown("### 📊 Health Metrics Ranges")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
<div class="card">
    <h4>📉 Blood Pressure</h4>
    <p><b>Normal:</b> <120/80 mmHg</p>
    <p><b>Elevated:</b> 120-129/<80 mmHg</p>
    <p><b>High Stage 1:</b> 130-139/80-89 mmHg</p>
    <p><b>High Stage 2:</b> ≥140/≥90 mmHg</p>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
<div class="card">
    <h4>📊 BMI Categories</h4>
    <p><b>Underweight:</b> <18.5</p>
    <p><b>Normal:</b> 18.5 - 24.9</p>
    <p><b>Overweight:</b> 25 - 29.9</p>
    <p><b>Obese:</b> ≥30</p>
</div>
""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
<div class="card">
    <h4>📈 Cholesterol Levels</h4>
    <p><b>Normal:</b> <200 mg/dL</p>
    <p><b>Borderline:</b> 200-239 mg/dL</p>
    <p><b>High:</b> ≥240 mg/dL</p>
    <p><b>Ideal LDL:</b> <100 mg/dL</p>
</div>
""", unsafe_allow_html=True)
    
    # Create a simple pie chart for risk distribution
    st.markdown("### 🎯 Risk Distribution in Population")
    
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    sizes = [60, 25, 15]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    col_pie1, col_pie2 = st.columns([1, 2])
    with col_pie1:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        ax2.set_title('Cardio Risk')
        st.pyplot(fig2)
    with col_pie2:
         st.write("The distribution shows that while the majority of the population maintains a low risk profile, significant portions fall into medium and high-risk categories, emphasizing the need for regular screenings.")

# -----------------------------
# Page 4: Prevention Tips
# -----------------------------
elif page == "💡 Prevention Tips":
    
    tabs = st.tabs(["🏃‍♂️ Lifestyle", "🍎 Diet", "📋 Monitoring", "🧘‍♀️ Stress Management"])
    
    with tabs[0]:
        st.markdown("""
<div class="card">
<h3>🏃‍♂️ Physical Activity Recommendations</h3>
<ul>
<li><b>Aerobic Exercise:</b> 150 minutes per week of moderate-intensity exercise</li>
<li><b>Strength Training:</b> 2 days per week focusing on major muscle groups</li>
<li><b>Daily Movement:</b> Take breaks from sitting every hour</li>
<li><b>Consistency:</b> Better to exercise regularly than intensely occasionally</li>
</ul>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("""
<div class="card">
<h3>🚭 Avoid Harmful Habits</h3>
<ul>
<li><b>Stop Smoking:</b> Reduces heart disease risk by 50% within 1 year</li>
<li><b>Limit Alcohol:</b> Maximum 1 drink per day for women, 2 for men</li>
<li><b>Avoid Sedentary Lifestyle:</b> Stand up and move every 30 minutes</li>
<li><b>Manage Stress:</b> Practice mindfulness and relaxation techniques</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
<div class="card">
<h3>🍎 Heart-Healthy Diet</h3>
<h4>✅ Foods to Include:</h4>
<ul>
<li><b>Fruits & Vegetables:</b> 5 servings per day minimum</li>
<li><b>Whole Grains:</b> Oats, brown rice, quinoa, whole wheat</li>
<li><b>Lean Protein:</b> Fish (especially salmon), skinless poultry, legumes</li>
<li><b>Healthy Fats:</b> Avocados, nuts, olive oil</li>
</ul>
<h4>❌ Foods to Limit:</h4>
<ul>
<li><b>Processed Foods:</b> High in sodium and preservatives</li>
<li><b>Trans Fats:</b> Found in fried foods and baked goods</li>
<li><b>Added Sugars:</b> Limit to less than 25g per day</li>
<li><b>Red Meat:</b> Choose lean cuts and limit consumption</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
<div class="card">
<h3>📋 Health Monitoring Schedule</h3>
<h4>Daily Monitoring:</h4>
<ul>
<li><b>Blood Pressure:</b> If you have hypertension</li>
<li><b>Physical Activity:</b> Aim for 10,000 steps</li>
<li><b>Weight:</b> Weekly monitoring recommended</li>
</ul>
<h4>Regular Checkups:</h4>
<ul>
<li><b>Annual Physical:</b> Complete health assessment</li>
<li><b>Cholesterol Test:</b> Every 4-6 years (more if high risk)</li>
<li><b>Blood Glucose:</b> Annual screening after age 45</li>
<li><b>ECG:</b> As recommended by your doctor</li>
</ul>
<h4>Warning Signs to Watch For:</h4>
<ul>
<li>Chest pain or discomfort</li>
<li>Shortness of breath</li>
<li>Irregular heartbeat</li>
<li>Excessive fatigue</li>
<li>Swelling in legs/ankles</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("""
<div class="card">
<h3>🧘‍♀️ Stress Management Techniques</h3>
<h4>Immediate Relief:</h4>
<ul>
<li><b>Deep Breathing:</b> 4-7-8 technique (inhale 4, hold 7, exhale 8)</li>
<li><b>Progressive Muscle Relaxation:</b> Tense and relax muscle groups</li>
<li><b>Mindful Walking:</b> 5-minute walk focusing on surroundings</li>
</ul>
<h4>Long-term Strategies:</h4>
<ul>
<li><b>Regular Exercise:</b> Natural stress reliever</li>
<li><b>Meditation:</b> 10 minutes daily using apps like Calm or Headspace</li>
<li><b>Sleep Hygiene:</b> 7-9 hours of quality sleep</li>
<li><b>Social Connections:</b> Maintain supportive relationships</li>
<li><b>Hobbies:</b> Engage in activities you enjoy</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Page 5: About Parameters
# -----------------------------
elif page == "📚 About Parameters":
    
    parameters = {
        "Age": {
            "description": "Risk increases with age as arteries become less flexible.",
            "normal_range": "N/A - Natural progression",
            "impact": "High - Uncontrollable but manageable",
            "tips": "Regular checkups become more important after 40"
        },
        "Blood Pressure": {
            "description": "Force of blood against artery walls. High BP damages arteries.",
            "normal_range": "<120/80 mmHg",
            "impact": "Very High - Major controllable factor",
            "tips": "Monitor regularly, reduce salt, manage stress"
        },
        "Cholesterol": {
            "description": "Waxy substance in blood. High levels form plaque in arteries.",
            "normal_range": "<200 mg/dL total cholesterol",
            "impact": "High - Builds up silently",
            "tips": "Limit saturated fats, eat soluble fiber, exercise"
        },
        "BMI (Body Mass Index)": {
            "description": "Measures body fat based on height and weight.",
            "normal_range": "18.5 - 24.9",
            "impact": "Medium - Indirect risk factor",
            "tips": "Combination of diet and exercise for healthy weight"
        },
        "Smoking": {
            "description": "Chemicals damage blood vessels and heart.",
            "normal_range": "Non-smoker",
            "impact": "Very High - #1 preventable cause",
            "tips": "Quit completely - benefits start immediately"
        },
        "Physical Activity": {
            "description": "Exercise strengthens heart and improves circulation.",
            "normal_range": "150 mins/week moderate exercise",
            "impact": "High - Protective factor",
            "tips": "Find activities you enjoy, consistency over intensity"
        },
        "Glucose Levels": {
            "description": "High blood sugar damages blood vessels over time.",
            "normal_range": "<100 mg/dL (fasting)",
            "impact": "High - Silent damage",
            "tips": "Limit refined carbs, maintain healthy weight"
        },
        "Alcohol": {
            "description": "Excessive drinking raises BP and adds calories.",
            "normal_range": "≤1 drink/day (women), ≤2 drinks/day (men)",
            "impact": "Medium - Dose-dependent",
            "tips": "Drink in moderation, have alcohol-free days"
        }
    }
    
    # Custom CSS for Parameter Cards
    st.markdown("""
    <style>
    .param-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #ccc;
        background: white;
        transition: transform 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .param-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .param-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .param-label {
        font-weight: 600;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

    # Define simple color scheme for borders
    colors = {
        "Age": "#3498db",          # Blue
        "Blood Pressure": "#e74c3c", # Red
        "Cholesterol": "#f1c40f",    # Yellow
        "BMI (Body Mass Index)": "#9b59b6", # Purple
        "Smoking": "#34495e",       # Dark
        "Physical Activity": "#2ecc71", # Green
        "Glucose Levels": "#e67e22", # Orange
        "Alcohol": "#d35400"        # Burnt Orange
    }

    for param, info in parameters.items():
        color = colors.get(param, "#95a5a6")
        
        st.markdown(f"""
        <div class="param-card" style="border-left-color: {color};">
            <div class="param-title" style="color: {color};">📌 {param}</div>
            <p>{info['description']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 10px;">
                <div><span class="param-label">Normal Range:</span> {info['normal_range']}</div>
                <div><span class="param-label">Impact:</span> {info['impact']}</div>
            </div>
            <div style="margin-top: 10px; background: rgba(0,0,0,0.02); padding: 10px; border-radius: 8px;">
                <span class="param-label">💡 Tip:</span> <i>{info['tips']}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Page 6: Model Analysis
# -----------------------------
elif page == "📈 Model Analysis":
    
    # Load dataset for visualization
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("data/cardio_train.csv", sep=";")
        except:
            # Fallback if delimiter is different or file missing
            return None

    df = load_data()
    
    if df is not None:
        tabs = st.tabs(["🔥 Correlation Matrix", "😵 Confusion Matrix", "✨ Feature Importance"])
        
        with tabs[0]:
            st.markdown("<div class='card'><h3>Feature Correlations</h3>", unsafe_allow_html=True)
            st.write("Understanding how different health factors relate to each other.")
            
            # Compute correlation
            corr = df.corr()
            
            # Plot
            # Plot
            # Reduced figure size for smaller display
            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, annot_kws={"size": 5})
            ax_corr.tick_params(axis='both', which='major', labelsize=5)
            # Adjust colorbar font size
            cbar = ax_corr.collections[0].colorbar
            cbar.ax.tick_params(labelsize=5)
            st.pyplot(fig_corr)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("<div class='card'><h3>Model Confusion Matrix</h3>", unsafe_allow_html=True)
            st.write("Evaluating model performance on a subset of data.")
            
            with st.spinner("Generating confusion matrix..."):
                try:
                    # Prepare data for prediction
                    # Using a subset for speed, but large enough for meaningful metrics
                    subset = df.sample(n=3000, random_state=42)
                    
                    # Initialize input dataframe with same columns as model features
                    X_subset = pd.DataFrame(index=subset.index)
                    
                    # 1. Age processing (days -> years)
                    X_subset['age_years'] = subset['age'] / 365.25
                    
                    # 2. Copy direct numeric columns
                    X_subset['height'] = subset['height']
                    X_subset['weight'] = subset['weight']
                    X_subset['ap_hi'] = subset['ap_hi']
                    X_subset['ap_lo'] = subset['ap_lo']
                    
                    # 3. Categorical (Ordinal/Binary) - copying distinct integer values
                    # Assuming dataset encoding matches model training encoding (1,2,3 / 0,1)
                    X_subset['cholesterol'] = subset['cholesterol']
                    X_subset['gluc'] = subset['gluc']
                    X_subset['smoke'] = subset['smoke']
                    X_subset['alco'] = subset['alco']
                    X_subset['active'] = subset['active']
                    
                    # 4. Gender One-Hot Encoding
                    # Dataset: 1=Female, 2=Male (Standard Cardio Train Dataset convention)
                    # We need to match feature_columns which likely has 'gender_Male' and 'gender_Female'
                    X_subset['gender_Male'] = (subset['gender'] == 2).astype(int)
                    X_subset['gender_Female'] = (subset['gender'] == 1).astype(int)
                    
                    # 5. Scaling
                    # Ensure we have all columns in correct order before scaling/predicting
                    # Use globally loaded 'num_cols' for scaling
                    X_subset[num_cols] = scaler.transform(X_subset[num_cols])
                    
                    # Reorder columns to match model expectation exactly
                    X_subset = X_subset[feature_columns]
                    
                    # Predict
                    y_pred = model.predict(X_subset)
                    y_true = subset['cardio']
                    
                    # Calculate Confusion Matrix
                    cm = confusion_matrix(y_true, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    
                    # Plot Heatmap
                    # Plot Heatmap
                    # Reduced figure size as requested to be very small
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 2))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=['Predicted Healthy', 'Predicted Disease'],
                                yticklabels=['Actual Healthy', 'Actual Disease'],
                                annot_kws={"size": 6})
                    ax_cm.set_title('Confusion Matrix', fontsize=7)
                    ax_cm.tick_params(axis='both', which='major', labelsize=5)
                    # Adjust colorbar font size
                    cbar = ax_cm.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=5)
                    st.pyplot(fig_cm)
                    
                    # Disclaimer about subset
                    st.caption("Metrics calculated on a random subset of 3,000 records from the training dataset.")

                except Exception as e:
                    st.error(f"Error calculating confusion matrix: {str(e)}")
                    st.warning("Please check column names and data format.")


            st.markdown("</div>", unsafe_allow_html=True)
            
        with tabs[2]:
            st.markdown("<div class='card'><h3>Feature Importance</h3>", unsafe_allow_html=True)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Align with feature names
                # We need features list
                feat_names = feature_columns
                
                fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
                sns.barplot(x=importances[indices], y=[feat_names[i] for i in indices], palette='magma', ax=ax_feat)
                ax_feat.set_title("Random Forest Feature Importance")
                st.pyplot(fig_feat)
            else:
                st.warning("Model does not support feature importance visualization.")
            
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("Could not load dataset for visualization. Please ensure 'data/cardio_train.csv' exists.")

# -----------------------------
# Page 7: About Project
# -----------------------------
elif page == "ℹ️ About Project":
    
    st.markdown("""
<div class="card">
    <h3>🎯 Project Overview</h3>
    <p>CardioCare is a machine learning-based application designed to assess 
    cardiovascular disease risk using clinical and lifestyle parameters. 
    The system provides personalized risk assessments and prevention recommendations.</p>
</div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
<div class="card">
<h3>🤖 Technology Stack</h3>
<ul>
<li><b>Frontend:</b> Streamlit</li>
<li><b>ML Framework:</b> Scikit-learn</li>
<li><b>Model:</b> Random Forest Classifier</li>
<li><b>Visualization:</b> Matplotlib</li>
<li><b>Data Processing:</b> Pandas, NumPy</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
<div class="card">
<h3>📊 Model Performance</h3>
<ul>
<li><b>Accuracy:</b> 73.53%</li>
<li><b>Precision:</b>  76.26%</li>
<li><b>Recall:</b> 68.09%</li>
<li><b>F1-Score:</b> 71.94%</li>
<li><b>AUC-ROC:</b> 0.80</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    # Data for the table
    comparison_data = {
        "Algorithm": [
            "Random Forest", 
            "Logistic Regression", 
            "Naive Bayes", 
            "SVM", 
            "Decision Tree"
        ],
        "Train Test Split": ["73.51%", "72.85%", "71.07%", "73.36%", "62.96%"],
        "K-Fold": ["70.24%", "72.94%", "71.22%", "73.47%", "63.99%"],
        "Hyperparameter Tuning": ["73.53%", "72.83%", "71.07%", "72.6%", "72.8%"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Convert to HTML with inline styling for simplicity and robustness
    table_html = comparison_df.to_html(index=False, border=0, classes=["custom-table"])
    
    # Styled Table HTML
    # We use a dedicated variable with no indentation to avoid Markdown code block interpretation
    html_code = f"""
<style>
.custom-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}
.custom-table th {{
    background-color: #f1f5f9;
    color: #475569;
    font-weight: 600;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #e2e8f0;
}}
.custom-table td {{
    padding: 12px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
    background-color: white; /* Ensure cells have white background */
}}
.custom-table tr:hover {{
    background-color: #f8fafc;
}}
</style>

<div class="card">
    <h3>🧪 Algorithm Comparison</h3>
    <p>Accuracy scores across different evaluation methods.</p>
    <div style="overflow-x: auto; background-color: white; border-radius: 8px;">
        {table_html}
    </div>
</div>
"""
    
    st.markdown(html_code, unsafe_allow_html=True)
    
    st.markdown("""
<div class="card">
<h3>📁 Dataset Information</h3>
<p>The model was trained on cardiovascular health data containing 70,000 records with 12 clinical features. The dataset includes balanced representation of various age groups and health conditions.</p>
<h4>Features Used:</h4>
<ul>
<li><b>Demographic:</b> Age, Gender, Height, Weight</li>
<li><b>Clinical:</b> Systolic BP, Diastolic BP, Cholesterol, Glucose</li>
<li><b>Lifestyle:</b> Smoking, Alcohol, Physical Activity</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="card">
<h3>⚠️ Important Disclaimer</h3>
<p>This application is for <b>educational and informational purposes only</b> 
and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
<p><b>Always seek the advice of your physician or qualified health provider</b> 
with any questions you may have regarding a medical condition.</p>
<p>The predictions are based on statistical models and may not be 100% accurate 
for all individuals. Use this tool as a preliminary assessment only.</p>
</div>
""", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
<div style='text-align: center;'>
<p>Made with ❤️ using Machine Learning & Streamlit</p>
<p><small>© 2024 CardioCare - Cardiovascular Health Assistant</small></p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Footer for all pages
# -----------------------------
# -----------------------------
# Footer for all pages
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(0,0,0,0.5);'>
    <p><small>For educational purposes only • Consult a doctor for medical advice</small></p>
</div>
""", unsafe_allow_html=True)