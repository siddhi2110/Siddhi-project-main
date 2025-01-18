import streamlit as st
from PIL import Image
from api import process_image, process_video
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-container {
            background-color: #f9f9f9;
            padding: 20px;
        }
        .title {
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.9em;
            margin-top: 20px;
        }
        .result {
            color: #6eb52f;
            font-size: 1.5em;
            font-weight: bold;
        }
        .result-fake {
            color: #ff4b4b;
            font-size: 1.5em;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Project Introduction
st.markdown('<div class="title">AI-Enhanced Deepfake Detection Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detecting Deepfake Images and Videos using AI-powered models</div>', unsafe_allow_html=True)

st.markdown(
    """
    ### About the Project
    In the age of AI-generated media, detecting deepfakes has become critical for ensuring the authenticity of visual content. This project leverages advanced deep learning models such as EfficientNet variations to analyze images and videos for deepfake detection. 
    
    **Key Features:**
    - Supports both image and video input.
    - Multiple deep learning models for analysis.
    - Threshold-based classification for better customization.
    - Interactive interface for seamless user experience.
    
    **How It Works:**
    1. Upload an image or video.
    2. Select the model and dataset preferences.
    3. Analyze the content and visualize the results.
    """
)

# Sidebar for File Upload and Configuration
with st.sidebar:
    st.header("Configuration")
    file_type = st.radio("Select File Type", ("Image", "Video"))
    uploaded_file = st.file_uploader(
        f"Upload a {file_type.lower()} file", type=["jpg", "jpeg", "png", "mp4"]
    )
    model = st.selectbox(
        "Select Model", (
            "EfficientNetB4",
            "EfficientNetB4ST",
            "EfficientNetAutoAttB4",
            "EfficientNetAutoAttB4ST",
        )
    )
    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    if file_type == "Video":
        frames = st.slider("Select Number of Frames to Analyze", 10, 100, 50)

# Main Area for Results
if uploaded_file:
    if file_type == "Image":
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.video(uploaded_file)

    if st.button("Analyze for Deepfake"):
        with st.spinner("Analyzing..."):
            if file_type == "Image":
                result, pred = process_image(
                    image=uploaded_file, model=model, dataset=dataset, threshold=threshold
                )
            else:
                video_path = f"uploads/{uploaded_file.name}"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())
                result, pred = process_video(
                    video_path, model=model, dataset=dataset, threshold=threshold, frames=frames
                )

        # Display Result
        st.markdown(
            f'<h3>The given {file_type} is: <span class="result-{"fake" if result == "fake" else "real"}"> {result.upper()} </span> with a probability of {pred:.2f}</h3>',
            unsafe_allow_html=True,
        )

        # Generate Graph using Plotly
        categories = ["Real", "Fake"]
        probabilities = [1 - pred, pred] if result == "fake" else [pred, 1 - pred]
        colors = ["#6eb52f", "#ff4b4b"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories, 
            y=probabilities, 
            marker_color=colors,
            text=[f"{p:.2%}" for p in probabilities],
            textposition='auto',
        ))
        fig.update_layout(
            title="Deepfake Detection Confidence",
            xaxis_title="Category",
            yaxis_title="Probability",
            template="plotly_dark",
        )
        st.plotly_chart(fig)

        # Additional Visualization Section
        st.markdown("### Model Efficiency and Accuracy")
        st.markdown(
            "This section represents the efficiency and accuracy of the selected model during deepfake detection.")

        # Mock Data for Visualization
        model_efficiency = {
            "EfficientNetB4": [90, 85],
            "EfficientNetB4ST": [92, 88],
            "EfficientNetAutoAttB4": [93, 89],
            "EfficientNetAutoAttB4ST": [94, 91],
        }
        labels = ["Efficiency (%)", "Accuracy (%)"]
        data = model_efficiency[model]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=labels, 
            y=data, 
            marker_color=["#4CAF50", "#2196F3"],
            text=[f"{d}%" for d in data],
            textposition='auto',
        ))
        fig2.update_layout(
            title=f"Performance Metrics of {model}",
            xaxis_title="Metric",
            yaxis_title="Percentage",
            template="plotly_dark",
        )
        st.plotly_chart(fig2)
else:
    st.info("Please upload a file to start the analysis.")

# Footer
st.markdown('<div class="footer">Built with Streamlit | Deepfake Detection Project</div>', unsafe_allow_html=True)
