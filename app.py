import streamlit as st
import tensorflow as tf
import numpy as np
import json
import sqlite3
import hashlib
import secrets
import re
from io import BytesIO
from datetime import datetime

from pathlib import Path
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as ReportLabImage, KeepTogether
)


# ================================================
# PAGE CONFIGURATION
# ================================================

st.set_page_config(
    page_title="AI Leaf Care and Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ================================================
# PATHS
# ================================================

MODEL_PATH = Path("model/best_leaf_model.keras")
CLASS_NAMES_PATH = Path("model/class_names.json")
DATABASE_PATH = Path("users.db")


# ================================================
# DATABASE AND AUTHENTICATION
# ================================================

def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()


def register_user(name, email, password):
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO users
                (name, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, password_hash, salt)
            )
        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(email, password):
    with sqlite3.connect(DATABASE_PATH) as connection:
        user = connection.execute(
            """
            SELECT name, password_hash, salt
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if user is None:
        return None

    name, stored_hash, salt = user
    entered_hash = hash_password(password, salt)

    if secrets.compare_digest(entered_hash, stored_hash):
        return name

    return None


initialize_database()


# ================================================
# SESSION MANAGEMENT
# ================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


# ================================================
# LOAD TRAINED MODEL
# ================================================

@st.cache_resource
def load_leaf_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found: model/best_leaf_model.keras"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Class names file not found: model/class_names.json"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "Number of class names does not match model outputs."
        )

    return model, class_names


# ================================================
# IMAGE PREPROCESSING
# ================================================

def preprocess_leaf_image(image):
    image = image.convert("RGB")
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# ================================================
# FORMAT MODEL PREDICTION
# ================================================

def format_prediction(label):
    parts = label.split("__", 1)

    if len(parts) == 2:
        plant = parts[0].capitalize()
        disease = parts[1]
    else:
        plant = "Unknown"
        disease = label

    if disease.startswith("Tomato___"):
        disease = disease.replace("Tomato___", "", 1)
    elif disease.startswith("Potato___"):
        disease = disease.replace("Potato___", "", 1)

    disease = disease.replace("_", " ")
    return plant, disease


# ================================================
# DISEASE INFORMATION
# ================================================

disease_information = {
    "anthracnose": {
        "symptoms": "Dark, irregular spots or lesions may appear on mango leaves.",
        "cause": "A fungal disease associated with Colletotrichum species.",
        "care": [
            "Remove severely affected fallen plant material.",
            "Avoid prolonged leaf wetness.",
            "Maintain good air circulation.",
            "Seek local agricultural advice if symptoms spread."
        ]
    },
    "bacterial canker": {
        "symptoms": "Dark lesions and canker-like symptoms may develop on affected mango plant parts.",
        "cause": "A bacterial disease associated with Xanthomonas species.",
        "care": [
            "Avoid unnecessary injury to the plant.",
            "Use clean gardening tools.",
            "Remove severely affected plant material appropriately.",
            "Consult an agricultural expert for confirmation."
        ]
    },
    "cutting weevil": {
        "symptoms": "Leaf damage may include irregular cuts or feeding-related injury.",
        "cause": "Feeding damage associated with cutting weevils.",
        "care": [
            "Inspect leaves and nearby plant parts regularly.",
            "Monitor the plant for insect activity.",
            "Remove heavily damaged plant material when appropriate.",
            "Seek local agricultural pest-management advice."
        ]
    },
    "die back": {
        "symptoms": "Affected shoots may show drying, discoloration and progressive dieback.",
        "cause": "Can be associated with fungal infection and plant stress.",
        "care": [
            "Remove affected branches using clean tools.",
            "Avoid unnecessary plant injuries.",
            "Maintain appropriate watering.",
            "Seek expert advice if symptoms continue."
        ]
    },
    "gall midge": {
        "symptoms": "Affected leaves may show abnormal growth, distortion or gall-like damage.",
        "cause": "Insect damage associated with gall midges.",
        "care": [
            "Inspect new leaves regularly.",
            "Monitor for insect activity.",
            "Remove severely affected plant material where appropriate.",
            "Consult a local agricultural specialist."
        ]
    },
    "powdery mildew": {
        "symptoms": "White or powder-like patches may appear on leaf surfaces.",
        "cause": "A fungal disease associated with powdery mildew fungi.",
        "care": [
            "Improve air circulation around the plant.",
            "Avoid overcrowding.",
            "Monitor new leaf growth.",
            "Seek appropriate local disease-management advice."
        ]
    },
    "sooty mould": {
        "symptoms": "Dark, black, soot-like growth may cover leaf surfaces.",
        "cause": "Often associated with honeydew produced by sap-feeding insects.",
        "care": [
            "Inspect the plant for sap-feeding insects.",
            "Address the underlying insect problem.",
            "Maintain plant cleanliness.",
            "Seek agricultural advice if the condition persists."
        ]
    },
    "early blight": {
        "symptoms": "Brown leaf spots may develop with concentric ring patterns.",
        "cause": "A fungal disease associated with Alternaria species.",
        "care": [
            "Remove severely affected fallen leaves.",
            "Avoid overhead watering where possible.",
            "Maintain good spacing and air circulation.",
            "Follow locally recommended disease-management practices."
        ]
    },
    "late blight": {
        "symptoms": "Water-soaked-looking lesions may develop and spread rapidly under favourable conditions.",
        "cause": "A destructive disease caused by Phytophthora infestans.",
        "care": [
            "Inspect nearby plants for similar symptoms.",
            "Avoid unnecessary leaf wetness.",
            "Remove affected plant material appropriately.",
            "Seek prompt agricultural advice because this disease can spread rapidly."
        ]
    },
    "bacterial spot": {
        "symptoms": "Small dark spots may appear on leaves, sometimes surrounded by yellow areas.",
        "cause": "A bacterial disease associated with Xanthomonas species.",
        "care": [
            "Avoid handling wet plants unnecessarily.",
            "Use clean gardening tools.",
            "Remove severely affected plant material appropriately.",
            "Seek local disease-management guidance."
        ]
    },
    "leaf mold": {
        "symptoms": "Yellowish areas may appear on upper leaf surfaces, with mould-like growth underneath.",
        "cause": "A fungal disease associated with Passalora fulva.",
        "care": [
            "Improve air circulation.",
            "Avoid excessive humidity.",
            "Remove affected leaves when appropriate.",
            "Monitor the plant regularly."
        ]
    },
    "septoria leaf spot": {
        "symptoms": "Numerous small circular spots may develop, sometimes with dark margins.",
        "cause": "A fungal disease associated with Septoria lycopersici.",
        "care": [
            "Remove affected fallen leaves.",
            "Avoid overhead watering.",
            "Keep adequate spacing between plants.",
            "Monitor disease progression."
        ]
    },
    "spider mites two-spotted spider mite": {
        "symptoms": "Tiny pale speckles, leaf discoloration and sometimes fine webbing may appear.",
        "cause": "Feeding damage caused by two-spotted spider mites.",
        "care": [
            "Inspect the undersides of leaves.",
            "Monitor the plant for mite activity.",
            "Maintain appropriate plant care.",
            "Seek locally suitable pest-management advice."
        ]
    },
    "target spot": {
        "symptoms": "Brown circular lesions with concentric rings may develop on leaves.",
        "cause": "A fungal disease associated with Corynespora cassiicola.",
        "care": [
            "Remove severely affected fallen leaves.",
            "Improve air circulation.",
            "Avoid prolonged leaf wetness.",
            "Monitor nearby plants."
        ]
    },
    "tomato mosaic virus": {
        "symptoms": "Leaves may show mottling, mosaic patterns and distortion.",
        "cause": "A viral disease caused by Tomato mosaic virus.",
        "care": [
            "Avoid handling plants unnecessarily.",
            "Clean tools between plants.",
            "Remove severely affected plants according to local guidance.",
            "Use healthy planting material."
        ]
    },
    "tomato yellow leaf curl virus": {
        "symptoms": "Leaves may curl upward, become yellow and show reduced growth.",
        "cause": "A viral disease transmitted by whiteflies.",
        "care": [
            "Monitor plants for whitefly activity.",
            "Remove severely affected plants where appropriate.",
            "Use healthy planting material.",
            "Seek local agricultural pest-management advice."
        ]
    },
    "healthy": {
        "symptoms": "The model predicts that the leaf belongs to the healthy category.",
        "cause": "No target disease was identified by the model.",
        "care": [
            "Continue appropriate watering.",
            "Provide suitable sunlight.",
            "Monitor the plant regularly.",
            "Maintain balanced plant nutrition."
        ]
    }
}


# ================================================
# CREATE DOWNLOADABLE PDF REPORT
# ================================================

def create_results_pdf(
    image,
    plant_name,
    disease_name,
    confidence,
    top_predictions,
    info
):
    """Create a PDF report in memory and return its bytes."""

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#176B3A"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#176B3A"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555")
    )

    story = []

    story.append(Paragraph(
        "AI LEAF CARE AND DISEASE DETECTION",
        title_style
    ))

    story.append(Paragraph(
        "AI-Assisted Leaf Analysis Report",
        subtitle_style
    ))

    story.append(Paragraph(
        f"<b>Report date:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        body_style
    ))

    # Add the uploaded image to the report.
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    report_image = ReportLabImage(
        image_buffer,
        width=75 * mm,
        height=65 * mm,
        kind="proportional"
    )
    story.append(Paragraph("Uploaded Leaf Image", section_style))
    story.append(report_image)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Prediction Results", section_style))

    result_data = [
        ["Predicted Plant", plant_name],
        ["Predicted Category", disease_name],
        ["Model Confidence", f"{confidence * 100:.2f}%"]
    ]

    result_table = Table(
        result_data,
        colWidths=[48 * mm, 112 * mm]
    )

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F4EC")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#222222")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9BC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(result_table)

    story.append(Paragraph("Top 3 Model Predictions", section_style))

    top_data = [["Plant", "Predicted Category", "Probability"]]
    for prediction in top_predictions:
        top_data.append([
            prediction["plant"],
            prediction["disease"],
            f'{prediction["score"] * 100:.2f}%'
        ])

    top_table = Table(
        top_data,
        colWidths=[40 * mm, 85 * mm, 35 * mm],
        repeatRows=1
    )

    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7C9BC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(top_table)

    story.append(Paragraph("Symptoms", section_style))
    story.append(Paragraph(info["symptoms"], body_style))

    story.append(Paragraph("Possible Cause", section_style))
    story.append(Paragraph(info["cause"], body_style))

    story.append(Paragraph("General Care Recommendations", section_style))
    for recommendation in info["care"]:
        story.append(Paragraph(
            f"&bull; {recommendation}",
            body_style
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Note: Model confidence is the model's output probability and "
        "is not a guarantee of diagnostic correctness.",
        small_style
    ))

    document.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ================================================
# CUSTOM DESIGN - ORIGINAL NEON UI
# ================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 15%, #163d20 0%, transparent 35%),
        radial-gradient(ellipse at 90% 85%, #182d1c 0%, transparent 35%),
        linear-gradient(135deg, #050d0a, #08151b, #090b13);
    color: white;
    font-family: 'Inter', sans-serif;
}

header, footer, #MainMenu {
    visibility: hidden;
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1450px;
}

.main-heading {
    text-align: center;
    color: #EFFF38;
    font-size: clamp(25px, 4vw, 42px);
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 0 0 8px #EFFF38, 0 0 18px #EFFF3880;
    margin-bottom: 40px;
}

.hero-title {
    color: #39FF88;
    font-size: clamp(32px, 4vw, 54px);
    font-weight: 800;
    line-height: 1.2;
    text-shadow: 0 0 8px #39FF88, 0 0 20px #39FF8880;
    margin-bottom: 20px;
}

.hero-description {
    color: #e0e9e3;
    font-size: 17px;
    line-height: 1.8;
    margin-bottom: 25px;
}

.section-heading {
    color: #EFFF38;
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 15px;
    text-shadow: 0 0 10px #EFFF3860;
}

.plant-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid #39FF8860;
    border-radius: 14px;
    padding: 15px 5px;
    text-align: center;
    color: white;
    font-size: 16px;
    font-weight: 700;
}

.explore-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #39FF88;
    border-radius: 8px;
    padding: 13px 15px;
    margin: 10px 0;
    color: #ffffff;
    font-size: 15px;
    font-weight: 600;
}

.st-key-login_card {
    background: rgba(8, 15, 17, 0.96);
    border: 2px solid #EFFF38;
    border-radius: 25px;
    padding: 28px;
    box-shadow:
        inset 0 0 18px rgba(255,255,255,0.18),
        0 0 12px #EFFF38,
        0 0 35px rgba(239,255,56,0.25);
}

.welcome {
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd6ce;
    font-size: 14px;
    margin-bottom: 20px;
}

.stTextInput label {
    color: white !important;
}

.stTextInput input {
    background: #111c24 !important;
    color: white !important;
    border: 1px solid #74808b !important;
    border-radius: 30px !important;
    min-height: 45px;
}

.stTextInput input:focus {
    border: 1px solid #39FF88 !important;
    box-shadow: 0 0 12px #39FF8860 !important;
}

.stFormSubmitButton button {
    width: 100%;
    min-height: 46px;
    border-radius: 30px;
    border: 2px solid white !important;
    background: linear-gradient(90deg, #ff4fa3, #efff38) !important;
    color: #101010 !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    box-shadow: 0 0 14px #ff4fa380, 0 0 18px #efff3860;
}

.stTabs [data-baseweb="tab-list"] {
    display: flex;
    width: 100%;
    gap: 10px;
    background: transparent;
    border-bottom: none !important;
}

.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    background: #111c24 !important;
    color: white !important;
    border: 1px solid #52615a !important;
    border-radius: 30px !important;
    padding: 12px 8px !important;
    font-weight: 800 !important;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(57, 255, 136, 0.12) !important;
    color: #39FF88 !important;
    border: 2px solid #39FF88 !important;
    box-shadow:
        0 0 8px #39FF88,
        0 0 18px #39FF8880,
        inset 0 0 10px #39FF8840 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.footer {
    text-align: center;
    color: #9eafa4;
    font-size: 12px;
    margin-top: 25px;
    line-height: 1.8;
}

.analysis-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid #39FF8870;
    border-radius: 18px;
    padding: 22px;
    margin-top: 15px;
}

div[data-testid="stFileUploader"] {
    background: rgba(57,255,136,0.04);
    border: 1px dashed #39FF88;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stFileUploader"] label {
    color: #EFFF38 !important;
    font-weight: 700;
}

.stButton button {
    border-radius: 30px;
    border: 1px solid #39FF88;
    background: linear-gradient(90deg, #ff4fa3, #efff38);
    color: #101010;
    font-weight: 800;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# ================================================
# MAIN PROJECT HEADING
# ================================================

st.markdown("""
<div class="main-heading">
    🌿 AI LEAF CARE AND DISEASE DETECTION 🌿
</div>
""", unsafe_allow_html=True)


# ================================================
# LANDING PAGE - BEFORE LOGIN
# ================================================

if not st.session_state.authenticated:

    left_column, right_column = st.columns(
        [1.25, 0.85],
        gap="large"
    )

    # --------------------------------
    # LEFT SIDE - PLANT INFORMATION
    # --------------------------------

    with left_column:

        st.markdown("""
        <div class="hero-title">
            Explore Plant Health 🌿
        </div>

        <div class="hero-description">
            A smart plant care platform designed to help
            identify leaf health concerns using Artificial
            Intelligence and Deep Learning.
        </div>

        <div class="section-heading">
            🌿 Plants That Can Be Identified
        </div>
        """, unsafe_allow_html=True)

        plant1, plant2, plant3 = st.columns(3)

        with plant1:
            st.markdown("""
            <div class="plant-box">
                🍅<br>Tomato
            </div>
            """, unsafe_allow_html=True)

        with plant2:
            st.markdown("""
            <div class="plant-box">
                🥔<br>Potato
            </div>
            """, unsafe_allow_html=True)

        with plant3:
            st.markdown("""
            <div class="plant-box">
                🥭­<br>Mango
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-heading">
            What You Can Explore
        </div>

        <div class="explore-item">
            1. Leaf Image Analysis
        </div>

        <div class="explore-item">
            2. Disease Identification
        </div>

        <div class="explore-item">
            3. Symptoms and Possible Causes
        </div>

        <div class="explore-item">
            4. Plant Care Recommendations
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------
    # RIGHT SIDE - LOGIN AND REGISTER
    # --------------------------------

    with right_column:

        with st.container(key="login_card"):

            st.markdown("""
            <div class="welcome">
                Welcome Back!
            </div>

            <div class="subtitle">
                Sign in to keep your plants healthy 🌿
            </div>
            """, unsafe_allow_html=True)

            login_tab, register_tab = st.tabs(
                ["LOGIN", "CREATE ACCOUNT"]
            )

            # --------------------------------
            # LOGIN
            # --------------------------------

            with login_tab:

                with st.form("login_form"):

                    email = st.text_input(
                        "Email address",
                        placeholder="Enter your email"
                    )

                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password"
                    )

                    login = st.form_submit_button("LOGIN")

                    if login:

                        if not email.strip() or not password:
                            st.error(
                                "Please enter your email and password."
                            )

                        else:

                            user_name = authenticate_user(
                                email.strip().lower(),
                                password
                            )

                            if user_name:
                                st.session_state.authenticated = True
                                st.session_state.user_name = user_name
                                st.success("Login successful!")
                                st.rerun()

                            else:
                                st.error(
                                    "Invalid email or password."
                                )

            # --------------------------------
            # CREATE ACCOUNT
            # --------------------------------

            with register_tab:

                with st.form("register_form"):

                    name = st.text_input(
                        "Full name",
                        placeholder="Enter your name"
                    )

                    new_email = st.text_input(
                        "Email address",
                        placeholder="Enter your email"
                    )

                    new_password = st.text_input(
                        "Create password",
                        type="password",
                        placeholder="Create a password"
                    )

                    confirm_password = st.text_input(
                        "Confirm password",
                        type="password",
                        placeholder="Confirm your password"
                    )

                    register = st.form_submit_button(
                        "CREATE ACCOUNT"
                    )

                    if register:

                        if (
                            not name.strip()
                            or not new_email.strip()
                            or not new_password
                            or not confirm_password
                        ):
                            st.error(
                                "Please fill in all the fields."
                            )

                        elif not re.fullmatch(
                            r"[^@\s]+@[^@\s]+\.[^@\s]+",
                            new_email.strip()
                        ):
                            st.error(
                                "Please enter a valid email address."
                            )

                        elif len(new_password) < 8:
                            st.error(
                                "Password must contain at least 8 characters."
                            )

                        elif new_password != confirm_password:
                            st.error(
                                "Passwords do not match."
                            )

                        else:

                            success, message = register_user(
                                name.strip(),
                                new_email.strip().lower(),
                                new_password
                            )

                            if success:
                                st.success(
                                    "Account created successfully! "
                                    "Please go to LOGIN and sign in."
                                )
                            else:
                                st.error(message)

            st.markdown("""
            <div class="footer">
                GROW SMARTER |· LIVE HEALTHIER<br>
                AI LEAF CARE AND DISEASE DETECTION 🌿
            </div>
            """, unsafe_allow_html=True)


# ================================================
# AI DASHBOARD - AFTER LOGIN
# ================================================

else:

    # --------------------------------
    # DASHBOARD HEADER
    # --------------------------------

    welcome_column, logout_column = st.columns([4, 1])

    with welcome_column:
        st.markdown(
            f"""
            <div class="hero-title">
                Welcome, {st.session_state.user_name}! 🌿
            </div>
            <div class="hero-description">
                Your plant health dashboard is ready.
                Upload a leaf image to analyse its health.
            </div>
            """,
            unsafe_allow_html=True
        )

    with logout_column:
        if st.button("LOGOUT", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.rerun()

    st.markdown("---")

    # --------------------------------
    # AI LEAF ANALYSIS SECTION
    # --------------------------------

    st.markdown("""
    <div class="section-heading">
        🌿 AI LEAF DISEASE ANALYSIS
    </div>

    <div class="hero-description">
        Upload a clear photograph of a tomato, potato or mango leaf
        to receive an AI-generated prediction.
    </div>
    """, unsafe_allow_html=True)

    upload_column, preview_column = st.columns([1, 1], gap="large")

    with upload_column:

        uploaded_file = st.file_uploader(
            "Upload your leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a single leaf."
        )

        predict_button = st.button(
            "🌿 ANALYSE LEAF IMAGE",
            use_container_width=True
        )

    with preview_column:

        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    uploaded_image,
                    caption="Uploaded Leaf Image",
                    use_container_width=True
                )

            except Exception:
                st.error(
                    "Unable to open this image. Please upload another image."
                )

    # --------------------------------
    # RUN MODEL PREDICTION
    # --------------------------------

    if predict_button:

        if uploaded_file is None:
            st.warning(
                "Please upload a leaf image first."
            )

        else:

            try:

                with st.spinner(
                    "AI model is analysing your leaf image..."
                ):

                    model, class_names = load_leaf_model()

                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file).convert("RGB")

                    processed_image = preprocess_leaf_image(image)

                    probabilities = model.predict(
                        processed_image,
                        verbose=0
                    )[0]

                    predicted_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_index])
                    predicted_label = class_names[predicted_index]

                    plant_name, disease_name = format_prediction(
                        predicted_label
                    )

                    top_indices = np.argsort(probabilities)[::-1][:3]

                # --------------------------------
                # ANALYSIS RESULTS
                # --------------------------------

                st.markdown("---")

                st.markdown("""
                <div class="section-heading">
                    🌿 ANALYSIS RESULTS
                </div>
                """, unsafe_allow_html=True)

                if confidence < 0.60:
                    st.warning(
                        "The model is uncertain about this prediction. "
                        "Please try a clearer leaf image and verify the result "
                        "with an agricultural expert."
                    )

                result_column1, result_column2 = st.columns(2)

                with result_column1:
                    st.markdown("### 🌿 Predicted Plant")
                    st.success(plant_name)

                with result_column2:
                    st.markdown("### 🌿 Predicted Category")
                    st.info(disease_name)

                # --------------------------------
                # MODEL CONFIDENCE
                # --------------------------------

                st.markdown("### 🌿 Model Confidence")
                st.progress(confidence)

                st.write(
                    f"Prediction confidence: **{confidence * 100:.2f}%**"
                )

                st.caption(
                    "Confidence is the model's output probability, "
                    "not a guarantee of diagnostic correctness."
                )

                # --------------------------------
                # TOP THREE PREDICTIONS
                # --------------------------------

                st.markdown("### Top 3 Model Predictions")

                top_predictions_for_pdf = []

                for index in top_indices:

                    top_plant, top_disease = format_prediction(
                        class_names[int(index)]
                    )

                    score = float(probabilities[int(index)])

                    top_predictions_for_pdf.append({
                        "plant": top_plant,
                        "disease": top_disease,
                        "score": score
                    })

                    st.write(
                        f"**{top_plant} - “ {top_disease}** "
                        f"({score * 100:.2f}%)"
                    )

                    st.progress(score)

                # --------------------------------
                # SYMPTOMS AND CARE
                # --------------------------------

                st.markdown("---")

                st.markdown("""
                <div class="section-heading">
                    🌿 PLANT HEALTH INFORMATION
                </div>
                """, unsafe_allow_html=True)

                info = disease_information.get(
                    disease_name.lower(),
                    {
                        "symptoms": "Refer to the predicted disease category.",
                        "cause": "The model identified this category based on learned image patterns.",
                        "care": [
                            "Inspect the plant carefully.",
                            "Monitor the affected leaves.",
                            "Consult a local agricultural expert."
                        ]
                    }
                )

                st.markdown("### Symptoms")
                st.write(info["symptoms"])

                st.markdown("### Possible Cause")
                st.write(info["cause"])

                st.markdown("### 🌿 General Care Recommendations")

                for recommendation in info["care"]:
                    st.markdown(f"- {recommendation}")

                # --------------------------------
                # DOWNLOAD RESULTS AS PDF
                # --------------------------------

                st.markdown("---")
                st.markdown("### 🌿“ Download Your Analysis Report")

                pdf_bytes = create_results_pdf(
                    image=image,
                    plant_name=plant_name,
                    disease_name=disease_name,
                    confidence=confidence,
                    top_predictions=top_predictions_for_pdf,
                    info=info
                )

                safe_plant_name = re.sub(
                    r"[^a-zA-Z0-9_-]",
                    "_",
                    plant_name
                )

                safe_disease_name = re.sub(
                    r"[^a-zA-Z0-9_-]",
                    "_",
                    disease_name
                )

                report_filename = (
                    f"leaf_analysis_{safe_plant_name}_"
                    f"{safe_disease_name}.pdf"
                )

                st.download_button(
                    label=" DOWNLOAD RESULTS AS PDF",
                    data=pdf_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Prediction could not be completed: {error}"
                )

                st.info(
                    "Check that the trained model and class_names.json "
                    "exist inside the model folder."
                )


# ================================================
# FOOTER
# ================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    🌿 AI LEAF CARE AND DISEASE DETECTION 🌿<br>
    Powered by MobileNetV2 | TensorFlow |Streamlit
</div>
""", unsafe_allow_html=True)