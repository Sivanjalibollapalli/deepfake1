import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime


class DeepfakeDetector:
    def __init__(self, api_key):
        genai.configure(api_key="AIzaSyBK4Ga3nUSwcuUHw_Jk8d74yK5_PE5xHWM")
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_image(self, image):
        prompt = """
        Analyze this image for signs of being AI-generated or manipulated (deepfake).

        Provide is it deepfake or real check deeply
        """

        response = self.model.generate_content([prompt, image])

        analysis = {
            'is_deepfake': self._interpret_response(response.text),
            'confidence_score': self._calculate_confidence(response.text)
        }

        return analysis

    def analyze_video_frame(self, frame, frame_number):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prompt = f"""
        Analyze frame {frame_number} of this video for signs of being AI-generated or manipulated.

        Provide a detailed analysis and confidence score.
        """

        response = self.model.generate_content([prompt, frame_pil])

        return {
            'frame_number': frame_number,
            'is_deepfake': self._interpret_response(response.text),
            'confidence_score': self._calculate_confidence(response.text),
        }

    def _interpret_response(self, response_text):
        keywords_deepfake = ['artificial', 'generated', 'fake', 'manipulated', 'unnatural']
        keywords_real = ['natural', 'authentic', 'genuine', 'real']

        deepfake_score = sum(1 for keyword in keywords_deepfake if keyword.lower() in response_text.lower())
        real_score = sum(1 for keyword in keywords_real if keyword.lower() in response_text.lower())

        return deepfake_score > real_score

    def _calculate_confidence(self, response_text):
        confidence_indicators = {
            'highly confident': 0.9,
            'confident': 0.7,
            'likely': 0.6,
            'possibly': 0.4,
            'uncertain': 0.3
        }

        for indicator, score in confidence_indicators.items():
            if indicator in response_text.lower():
                return score

        return 0.5


def main():
    st.set_page_config(
        page_title="Deepfake Detection System",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Deepfake Detection System")
    st.write("Upload an image or video to analyze for potential deepfake manipulation")



    # Initialize detector
    try:
        detector = DeepfakeDetector(api_key="AIzaSyBK4Ga3nUSwcuUHw_Jk8d74yK5_PE5xHWM")
    except Exception as e:
        st.error(f"Error initializing the detector: {str(e)}")
        return

    # File upload
    file_type = st.radio("Select file type:", ["Image", "Video"])
    uploaded_file = st.file_uploader(
        f"Choose a {file_type.lower()}",
        type=['jpg', 'jpeg', 'png'] if file_type == "Image" else ['mp4', 'avi', 'mov']
    )

    if uploaded_file:
        try:
            if file_type == "Image":
                analyze_image(uploaded_file, detector)
            else:
                analyze_video(uploaded_file, detector)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def analyze_image(uploaded_file, detector):
    # Display original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing image..."):
            results = detector.analyze_image(image)
            result_color = "red" if results['is_deepfake'] else "green"
            verdict = "Potential Deepfake" if results['is_deepfake'] else "Likely Authentic"
            st.markdown(f"""
            ### Verdict: <span style='color:{result_color}'>{verdict}</span>
            """, unsafe_allow_html=True)

            st.progress(results['confidence_score'])
            st.write(f"Confidence Score: {results['confidence_score']:.2%}")



def analyze_video(uploaded_file, detector):
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Video analysis setup
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Display video info
    st.subheader("Video Information")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"FPS: {fps}")

    # Analysis settings
    sample_rate = st.slider("Frame Sample Rate", 1, fps, fps // 2)

    if st.button("Start Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_results = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                status_text.text(f"Analyzing frame {frame_count}/{total_frames}")

                # Analyze frame
                result = detector.analyze_video_frame(frame, frame_count)
                frame_results.append(result)

                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)

            frame_count += 1

        cap.release()
        os.unlink(video_path)

        # Display results
        st.subheader("Analysis Results")

        deepfake_frames = sum(1 for frame in frame_results if frame['is_deepfake'])
        avg_confidence = sum(frame['confidence_score'] for frame in frame_results) / len(frame_results)

        st.markdown(f"""
        ### Summary
        - Analyzed Frames: {len(frame_results)}
        - Suspected Deepfake Frames: {deepfake_frames}
        - Average Confidence Score: {avg_confidence:.2%}
        - Overall Assessment: {"Likely Deepfake" if deepfake_frames > len(frame_results) / 2 else "Likely Authentic"}
        """)

        with st.expander("Frame-by-Frame Analysis", expanded=False):
            for result in frame_results:
                st.markdown(f"""
                #### Frame {result['frame_number']}
                - Verdict: {"Potential Deepfake" if result['is_deepfake'] else "Likely Authentic"}
                - Confidence: {result['confidence_score']:.2%}
                ---
                """)


if __name__ == "__main__":
    main()