import streamlit as st
import tensorflow as tf
import numpy as np
import json
import sqlite3
import hashlib
import secrets
import re
import html
import matplotlib.pyplot as plt

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
    Image as ReportLabImage
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

        connection.execute("""
            CREATE TABLE IF NOT EXISTS plant_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plant_name TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_name TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
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
            SELECT id, name, password_hash, salt
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if user is None:
        return None

    user_id, name, stored_hash, salt = user
    entered_hash = hash_password(password, salt)

    if secrets.compare_digest(entered_hash, stored_hash):

        return {
            "id": user_id,
            "name": name
        }

    return None


initialize_database()


# ================================================
# PLANT HEALTH HISTORY
# ================================================

def save_plant_history(
    user_id,
    plant_name,
    disease_name,
    confidence,
    image_name
):

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            INSERT INTO plant_history (
                user_id,
                plant_name,
                disease_name,
                confidence,
                image_name,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                plant_name,
                disease_name,
                confidence,
                image_name,
                current_time
            )
        )


def get_plant_history(user_id):

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.row_factory = sqlite3.Row

        records = connection.execute(
            """
            SELECT
                id,
                plant_name,
                disease_name,
                confidence,
                image_name,
                created_at
            FROM plant_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,)
        ).fetchall()

    return [dict(record) for record in records]


# ================================================
# SESSION MANAGEMENT
# ================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "login_success" not in st.session_state:
    st.session_state.login_success = False


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

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return np.expand_dims(image_array, axis=0)


# ================================================
# FORMAT MODEL PREDICTION
# ================================================

def format_prediction(label):

    label = str(label).strip()

    if "___" in label:

        plant, disease = label.split("___", 1)

    elif "__" in label:

        plant, disease = label.split("__", 1)

    else:

        plant = "Unknown"
        disease = label

    plant = plant.strip().capitalize()
    disease = disease.replace("_", " ").strip()

    return plant, disease


# ================================================
# FORMAT OLD HISTORY DISEASE NAMES
# ================================================

def format_history_disease(disease_name):

    raw = str(disease_name).strip()

    if not raw:
        return "Unknown"

    if "___" in raw:

        disease = raw.split("___", 1)[1]

    elif "__" in raw:

        disease = raw.split("__", 1)[1]

    else:

        disease = raw

    disease = disease.replace("_", " ").strip()

    if disease.lower() == "healthy":
        return "Healthy"

    return disease.title()


# ================================================
# GRAD-CAM MODEL
# ================================================

@st.cache_resource
def load_gradcam_model():

    model, _ = load_leaf_model()

    backbone = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if len(layer.output.shape) == 4:
                backbone = layer
                break

    if backbone is None:
        raise ValueError(
            "MobileNetV2 backbone not found inside the trained model."
        )

    try:

        last_conv_layer = backbone.get_layer("out_relu")

    except ValueError:

        raise ValueError(
            "The final MobileNetV2 layer 'out_relu' was not found."
        )

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output
    )

    gap_layer = None
    dropout_layer = None
    dense_layer = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D
        ):
            gap_layer = layer

        elif isinstance(
            layer,
            tf.keras.layers.Dropout
        ):
            dropout_layer = layer

        elif isinstance(
            layer,
            tf.keras.layers.Dense
        ):
            dense_layer = layer

    if (
        gap_layer is None
        or dropout_layer is None
        or dense_layer is None
    ):
        raise ValueError(
            "Could not find the original classifier layers."
        )

    return (
        feature_extractor,
        gap_layer,
        dropout_layer,
        dense_layer
    )


# ================================================
# GENERATE GRAD-CAM HEATMAP
# ================================================

def generate_gradcam(image, class_index):

    (
        feature_extractor,
        gap_layer,
        dropout_layer,
        dense_layer
    ) = load_gradcam_model()

    processed_image = preprocess_leaf_image(image)

    with tf.GradientTape() as tape:

        feature_maps = feature_extractor(
            processed_image,
            training=False
        )

        x = gap_layer(feature_maps)

        x = dropout_layer(
            x,
            training=False
        )

        predictions = dense_layer(x)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        feature_maps[0] * weights[0],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap_max = tf.reduce_max(heatmap)

    if float(heatmap_max.numpy()) <= 1e-8:
        raise ValueError(
            "Grad-CAM produced an empty heatmap."
        )

    heatmap = heatmap / heatmap_max

    heatmap_array = heatmap.numpy()

    original_image = image.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(heatmap_array * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    colour_map = plt.get_cmap("jet")

    coloured_heatmap = np.uint8(
        colour_map(heatmap_array)[:, :, :3] * 255
    )

    coloured_heatmap_image = Image.fromarray(
        coloured_heatmap
    )

    overlay_image = Image.blend(
        original_image,
        coloured_heatmap_image,
        alpha=0.40
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.imshow(original_image)

    axis.imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.50,
        vmin=0,
        vmax=1
    )

    axis.axis("off")
    figure.tight_layout(pad=0)

    heatmap_buffer = BytesIO()

    figure.savefig(
        heatmap_buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=150
    )

    plt.close(figure)

    heatmap_buffer.seek(0)

    heatmap_visualization = Image.open(
        heatmap_buffer
    ).convert("RGB")

    return (
        heatmap_visualization,
        overlay_image
    )


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

    image_buffer = BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="JPEG",
        quality=90
    )

    image_buffer.seek(0)

    report_image = ReportLabImage(
        image_buffer,
        width=75 * mm,
        height=65 * mm,
        kind="proportional"
    )

    story.append(Paragraph(
        "Uploaded Leaf Image",
        section_style
    ))

    story.append(report_image)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Prediction Results",
        section_style
    ))

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

    story.append(Paragraph(
        "Top 3 Model Predictions",
        section_style
    ))

    top_data = [
        ["Plant", "Predicted Category", "Probability"]
    ]

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

    story.append(Paragraph(
        "General Care Recommendations",
        section_style
    ))

    for recommendation in info["care"]:

        story.append(Paragraph(
            f"&bull; {recommendation}",
            body_style
        ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Note: Model confidence is the model's output probability "
        "and is not a guarantee of diagnostic correctness.",
        small_style
    ))

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ================================================
# CUSTOM DESIGN - NEON UI
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

/* LOGIN CARD: DARK BACKGROUND + NEON YELLOW BORDER */

.st-key-login_card {
    background: #0B1511 !important;
    border: 2px solid #EFFF38 !important;
    border-radius: 25px;
    padding: 28px;
    box-shadow:
        0 0 12px #EFFF38,
        0 0 25px rgba(239,255,56,0.45),
        0 0 40px rgba(239,255,56,0.20);
}

.st-key-login_card p,
.st-key-login_card label,
.st-key-login_card [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
}

.st-key-login_card .welcome {
    text-align: center;
    color: #EFFF38 !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.st-key-login_card .subtitle {
    text-align: center;
    color: #D0DCCF !important;
    font-size: 14px;
    margin-bottom: 20px;
}

.st-key-login_card .stTextInput input {
    background: #17251E !important;
    color: #FFFFFF !important;
    border: 1px solid #EFFF38 !important;
    border-radius: 30px !important;
    min-height: 45px;
}

.st-key-login_card .stTextInput input::placeholder {
    color: #AAB8AC !important;
}

.st-key-login_card .stTabs [data-baseweb="tab"] {
    background: #17251E !important;
    color: #FFFFFF !important;
    border: 1px solid #66745D !important;
}

.st-key-login_card .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #29351B !important;
    color: #EFFF38 !important;
    border: 2px solid #EFFF38 !important;
}

.st-key-login_card .footer {
    color: #C0CBBF !important;
}

/* GENERAL INPUTS */

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

/* TABS */

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

/* HISTORY CARDS: EXTRA DARK TEXT */

.history-card {
    background: #FFFFFF !important;
    border: 1px solid #A6B9AA;
    border-left: 5px solid #176B3A;
    border-radius: 12px;
    padding: 17px 20px;
    margin: 12px 0;
    color: #071207 !important;
}

.history-card * {
    color: #071207 !important;
}

.history-card .history-title {
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #071207 !important;
}

.history-card .history-detail {
    font-size: 14px;
    margin: 5px 0;
    color: #101A13 !important;
}

.history-card .history-confidence {
    font-size: 15px;
    font-weight: 800;
    color: #10552B !important;
}
/* ACCOUNT CREATED SUCCESS MESSAGE */

.st-key-login_card [data-testid="stAlert"] {
    background-color: #DFF5D8 !important;
    border: 1px solid #8BCF91 !important;
    border-radius: 10px !important;
}

.st-key-login_card [data-testid="stAlert"] p,
.st-key-login_card [data-testid="stAlert"] span,
.st-key-login_card [data-testid="stAlert"] div {
    color: #12351F !important;
    font-weight: 700 !important;
}

.st-key-login_card [data-testid="stAlert"] svg {
    fill: #176B3A !important;
    color: #176B3A !important;
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
                🥭<br>Mango
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

        <div class="explore-item">
            5. AI Explainability with Grad-CAM
        </div>

        <div class="explore-item">
            6. Plant Health History
        </div>
        """, unsafe_allow_html=True)

    # LOGIN AND REGISTER

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

            # LOGIN

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

                            user = authenticate_user(
                                email.strip().lower(),
                                password
                            )

                            if user:

                                st.session_state.authenticated = True
                                st.session_state.user_name = user["name"]
                                st.session_state.user_id = user["id"]
                                st.session_state.last_analysis = None
                                st.session_state.login_success = True

                                st.rerun()

                            else:

                                st.error(
                                    "Invalid email or password."
                                )

            # CREATE ACCOUNT

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
                GROW SMARTER | LIVE HEALTHIER<br>
                AI LEAF CARE AND DISEASE DETECTION 🌿
            </div>
            """, unsafe_allow_html=True)


# ================================================
# AI DASHBOARD - AFTER LOGIN
# ================================================

else:

    # SHOW LOGIN SUCCESS MESSAGE AFTER RERUN

    if st.session_state.login_success:

        st.markdown(
            """
            <div style="
                background-color: #DFF5D8;
                color: #12351F;
                border: 1px solid #A5D6A7;
                padding: 12px 16px;
                border-radius: 8px;
                font-weight: 700;
                ">
                Login successful! Welcome to your dashboard.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.login_success = False

    welcome_column, logout_column = st.columns([4, 1])

    with welcome_column:

        safe_user_name = html.escape(
            st.session_state.user_name
        )

        st.markdown(
            f"""
            <div class="hero-title">
                Welcome, {safe_user_name}! 🌿
            </div>

            <div class="hero-description">
                Your plant health dashboard is ready.
                Upload a leaf image to analyse its health.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================================================
    # LOGOUT
    # ================================================

    with logout_column:

        if st.button("LOGOUT", use_container_width=True):

            st.session_state.authenticated = False
            st.session_state.user_name = ""
            st.session_state.user_id = None
            st.session_state.last_analysis = None
            st.session_state.login_success = False

            st.rerun()

    st.markdown("---")

    analysis_tab, history_tab = st.tabs(
        [
            "🌿 LEAF ANALYSIS",
            "📊 PLANT HEALTH HISTORY"
        ]
    )

    # ================================================
    # TAB 1 - LEAF ANALYSIS
    # ================================================

    with analysis_tab:

        st.markdown("""
        <div class="section-heading">
            🌿 AI LEAF DISEASE ANALYSIS
        </div>

        <div class="hero-description">
            Upload a clear photograph of a tomato, potato or mango leaf
            to receive an AI-generated prediction.
        </div>
        """, unsafe_allow_html=True)

        upload_column, preview_column = st.columns(
            [1, 1],
            gap="large"
        )

        with upload_column:

            uploaded_file = st.file_uploader(
                "Upload your leaf image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a single leaf.",
                key="leaf_image_uploader"
            )

            predict_button = st.button(
                "🌿 ANALYSE LEAF IMAGE",
                use_container_width=True
            )

        with preview_column:

            if uploaded_file is not None:

                try:

                    preview_image = Image.open(
                        BytesIO(uploaded_file.getvalue())
                    ).convert("RGB")

                    st.image(
                        preview_image,
                        caption="Uploaded Leaf Image",
                        width=300
                    )

                except Exception:

                    st.error(
                        "Unable to open this image. Please upload another image."
                    )

        # RUN MODEL PREDICTION

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

                        image = Image.open(
                            BytesIO(uploaded_file.getvalue())
                        ).convert("RGB")

                        processed_image = preprocess_leaf_image(image)

                        probabilities = model.predict(
                            processed_image,
                            verbose=0
                        )[0]

                        predicted_index = int(
                            np.argmax(probabilities)
                        )

                        confidence = float(
                            probabilities[predicted_index]
                        )

                        predicted_label = class_names[predicted_index]

                        plant_name, disease_name = format_prediction(
                            predicted_label
                        )

                        top_indices = np.argsort(
                            probabilities
                        )[::-1][:3]

                        top_predictions_for_pdf = []

                        for index in top_indices:

                            top_plant, top_disease = format_prediction(
                                class_names[int(index)]
                            )

                            score = float(
                                probabilities[int(index)]
                            )

                            top_predictions_for_pdf.append({
                                "plant": top_plant,
                                "disease": top_disease,
                                "score": score
                            })

                        info = disease_information.get(
                            disease_name.lower(),
                            {
                                "symptoms": (
                                    "Refer to the predicted disease category."
                                ),
                                "cause": (
                                    "The model identified this category "
                                    "based on learned image patterns."
                                ),
                                "care": [
                                    "Inspect the plant carefully.",
                                    "Monitor the affected leaves.",
                                    "Consult a local agricultural expert."
                                ]
                            }
                        )

                        save_plant_history(
                            user_id=st.session_state.user_id,
                            plant_name=plant_name,
                            disease_name=disease_name,
                            confidence=confidence,
                            image_name=uploaded_file.name
                        )

                        # GENERATE GRAD-CAM

                        gradcam_overlay = None
                        gradcam_heatmap = None
                        gradcam_error = None

                        try:

                            (
                                gradcam_heatmap,
                                gradcam_overlay
                            ) = generate_gradcam(
                                image,
                                predicted_index
                            )

                        except Exception as error:

                            gradcam_error = str(error)

                        st.session_state.last_analysis = {
                            "image": image,
                            "plant_name": plant_name,
                            "disease_name": disease_name,
                            "confidence": confidence,
                            "predicted_index": predicted_index,
                            "top_predictions": top_predictions_for_pdf,
                            "info": info,
                            "gradcam_heatmap": gradcam_heatmap,
                            "gradcam_overlay": gradcam_overlay,
                            "gradcam_error": gradcam_error,
                            "filename": uploaded_file.name
                        }

                    st.success(
                        "Leaf analysis completed successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"Prediction could not be completed: {error}"
                    )

                    st.info(
                        "Check that the trained model and class_names.json "
                        "exist inside the model folder."
                    )

        # DISPLAY SAVED ANALYSIS

        analysis = st.session_state.last_analysis

        if analysis is not None:

            image = analysis["image"]
            plant_name = analysis["plant_name"]
            disease_name = analysis["disease_name"]
            confidence = analysis["confidence"]
            top_predictions_for_pdf = analysis["top_predictions"]
            info = analysis["info"]

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

            # MODEL CONFIDENCE

            st.markdown("### 🌿 Model Confidence")

            st.progress(confidence)

            st.write(
                f"Prediction confidence: **{confidence * 100:.2f}%**"
            )

            st.caption(
                "Confidence is the model's output probability, "
                "not a guarantee of diagnostic correctness."
            )

            # TOP THREE PREDICTIONS

            st.markdown("### Top 3 Model Predictions")

            for prediction in top_predictions_for_pdf:

                st.write(
                    f'**{prediction["plant"]} - '
                    f'{prediction["disease"]}** '
                    f'({prediction["score"] * 100:.2f}%)'
                )

                st.progress(prediction["score"])

            # GRAD-CAM EXPLAINABILITY

            st.markdown("---")

            st.markdown("""
            <div class="section-heading">
                🧠 AI EXPLAINABILITY - GRAD-CAM
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                """
                Grad-CAM creates a visual heatmap showing the image
                regions that influenced the model's prediction.
                """
            )

            if analysis["gradcam_overlay"] is not None:

                cam_column1, cam_column2 = st.columns(2)

                with cam_column1:

                    st.image(
                        image,
                        caption="Original Leaf Image",
                        width=300
                    )

                with cam_column2:

                    st.image(
                        analysis["gradcam_overlay"],
                        caption="Grad-CAM Highlighted Image",
                        width=300
                    )

                st.markdown("### 🔥 Grad-CAM Heatmap")

                st.image(
                    analysis["gradcam_heatmap"],
                    caption="Influential Regions Highlighted on the Leaf",
                    width=300
                )

                st.caption(
                    "The coloured regions indicate areas that influenced "
                    "the model's prediction. They should not be interpreted "
                    "as an exact measurement of diseased tissue."
                )

            else:

                st.warning(
                    "The prediction was successful, but Grad-CAM could "
                    "not be generated for this model."
                )

                if analysis["gradcam_error"]:

                    st.caption(
                        f"Grad-CAM detail: {analysis['gradcam_error']}"
                    )

            # SYMPTOMS AND CARE

            st.markdown("---")

            st.markdown("""
            <div class="section-heading">
                🌿 PLANT HEALTH INFORMATION
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Symptoms")
            st.write(info["symptoms"])

            st.markdown("### Possible Cause")
            st.write(info["cause"])

            st.markdown("### 🌿 General Care Recommendations")

            for recommendation in info["care"]:
                st.markdown(f"- {recommendation}")

            # DOWNLOAD RESULTS AS PDF

            st.markdown("---")

            st.markdown(
                "### 🌿 Download Your Analysis Report"
            )

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
                label="DOWNLOAD RESULTS AS PDF",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True
            )

    # ================================================
    # TAB 2 - PLANT HEALTH HISTORY
    # ================================================

    with history_tab:

        st.markdown("""
        <div class="section-heading">
            📊 YOUR PLANT HEALTH HISTORY
        </div>

        <div class="hero-description">
            View your previously analysed leaves and their prediction results.
        </div>
        """, unsafe_allow_html=True)

        history_records = get_plant_history(
            st.session_state.user_id
        )

        if not history_records:

            st.info(
                "No plant analysis history is available yet. "
                "Analyse your first leaf image to see it here."
            )

        else:

            total_analyses = len(history_records)

            healthy_count = sum(
                1 for record in history_records
                if format_history_disease(
                    record["disease_name"]
                ).lower() == "healthy"
            )

            other_count = total_analyses - healthy_count

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Total Analyses",
                    total_analyses
                )

            with metric2:

                st.metric(
                    "Healthy Predictions",
                    healthy_count
                )

            with metric3:

                st.metric(
                    "Other Predictions",
                    other_count
                )

            st.markdown("---")

            st.markdown("### Previous Analyses")

            for record in history_records:

                safe_plant = html.escape(
                    str(record["plant_name"]).strip().title()
                )

                safe_disease = html.escape(
                    format_history_disease(
                        record["disease_name"]
                    )
                )

                safe_image_name = html.escape(
                    str(record["image_name"] or "Not available")
                )

                safe_date = html.escape(
                    str(record["created_at"])
                )

                confidence_text = (
                    f'{record["confidence"] * 100:.2f}%'
                )

                st.markdown(
                    f"""
                    <div class="history-card">
                        <div class="history-title">
                            🌿 {safe_plant}
                        </div>

                        <div class="history-detail">
                            <b>Predicted category:</b> {safe_disease}
                        </div>

                        <div class="history-detail">
                            <b>Image:</b> {safe_image_name}
                        </div>

                        <div class="history-detail">
                            <b>Date:</b> {safe_date}
                        </div>

                        <div class="history-confidence">
                            Confidence: {confidence_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.caption(
                "Your history is displayed for the currently logged-in account."
            )


# ================================================
# FOOTER
# ================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    🌿 AI LEAF CARE AND DISEASE DETECTION 🌿<br>
    Powered by MobileNetV2 | TensorFlow | Streamlit
</div>
""", unsafe_allow_html=True)