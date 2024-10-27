import streamlit as st
import base64

def landing_page():
    st.set_page_config(layout="wide", page_title="AttendAI")

    # Custom CSS for styling
    st.markdown("""
    <style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 30px;
    }
    .medium-font {
        font-size: 25px !important;
        text-align: center;
        color: #4CAF50;  /* Changed to a green color for better visibility */
        margin-bottom: 50px;
    }
    .feature-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #1E88E5;
    }
    .feature-text {
        font-size: 16px !important;
        color: #4CAF50;  /* Changed to a green color for better visibility */
    }
    .stButton>button {
        font-size: 24px;
        padding: 15px 30px;
        border-radius: 10px;
        background-color: #1E88E5;
        color: white;
        display: block !important;
        margin: 0 auto !important;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="big-font">AI-Powered Attendance Management System</p>', unsafe_allow_html=True)

    # Subheader
    st.markdown('<p class="medium-font">Streamline your attendance management with our intelligent, automated system designed for modern organizations.</p>', unsafe_allow_html=True)

    # Features section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="feature-header">Facial Recognition</p>', unsafe_allow_html=True)
        st.markdown('<p class="feature-text">Accurately identify and mark attendance using advanced facial recognition technology.</p>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="feature-header">Real-time Tracking</p>', unsafe_allow_html=True)
        st.markdown('<p class="feature-text">Monitor attendance in real-time and generate instant reports for better decision-making.</p>', unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="feature-header">Easy Integration</p>', unsafe_allow_html=True)
        st.markdown('<p class="feature-text">Seamlessly integrate with existing systems for a smooth transition to automated attendance management.</p>', unsafe_allow_html=True)

    # Spacer
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Get Started button (centered)
    if st.button("Get Started"):
        st.session_state.page = "login_register"

    # Footer
    st.markdown("""
    <div class="footer">
        © 2023 AttendAI. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    landing_page()
