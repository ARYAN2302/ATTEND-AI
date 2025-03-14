**STILL UNDER DEVELOPMENT NOT READY FOR USE**( developmenting restarted date--12mar 2025)
# AttendAI - Ai Powered Attendance Management System

## Overview
AttendAI is an intelligent attendance management system that uses facial recognition technology to automate the attendance tracking process. The system captures attendance through photo upload, recognizing registered faces and maintaining attendance records efficiently.

## Features
- 👤 Real-time facial recognition
- 📊 Automated attendance tracking
- 💾 Database integration for attendance records
- 🔒 Secure face encoding storage
- 📱 Simple User-friendly interface

## Prerequisites
- Python 3.12 or higher
- macOS, Windows, or Linux operating system

## Installation

1. Clone the repository:
bash
git clone https://github.com/ARYAN2302/attendai.git
cd attendai


2. Create a virtual environment:

bash
python3 -m venv venv


3. Activate the virtual environment:
- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

4. Install required packages:
bash
pip install -r requirements.txt

## Usage
1. Run the main application:
uvicorn api:app --reload
2. Run the streamlit app:
streamlit run streamlit_app.py.
