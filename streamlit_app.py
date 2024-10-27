import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np
import base64
import logging
from landing_page import landing_page
import json
import face_recognition  # Added face_recognition import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"  # Update this if your API is hosted elsewhere

def save_face_encoding(student_id, face_encoding, face_image):
    try:
        with open('face_encodings.json', 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Create a dictionary structure for the student data
    student_data = {
        'encoding': face_encoding.tolist() if isinstance(face_encoding, np.ndarray) else face_encoding,
        'image': base64.b64encode(face_image.getvalue()).decode('utf-8')
    }
    
    # Store the student data
    data[str(student_id)] = student_data
    
    logger.info(f"Saving face encoding for student ID: {student_id}")

    with open('face_encodings.json', 'w') as file:
        json.dump(data, file)

def load_known_faces():
    try:
        with open('face_encodings.json', 'r') as file:
            data = json.load(file)
            logger.info(f"Loading data for {len(data)} students")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("No face encodings file found or invalid JSON")
        return [], [], []

    known_face_encodings = []
    known_face_ids = []
    known_face_images = []

    for student_id, student_data in data.items():
        try:
            # Ensure we have a dictionary with the required fields
            if isinstance(student_data, dict) and 'encoding' in student_data and 'image' in student_data:
                encoding = np.array(student_data['encoding'])
                image_data = base64.b64decode(student_data['image'])
                
                known_face_encodings.append(encoding)
                known_face_ids.append(student_id)
                known_face_images.append(image_data)
                logger.info(f"Successfully loaded data for student {student_id}")
            else:
                logger.warning(f"Invalid data structure for student {student_id}")
        except Exception as e:
            logger.error(f"Error loading face data for student {student_id}: {str(e)}")
            continue

    logger.info(f"Successfully loaded {len(known_face_encodings)} face encodings")
    return known_face_encodings, known_face_ids, known_face_images

def init_session_state():
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'auth_type' not in st.session_state:
        st.session_state.auth_type = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'dashboard'
    if 'refresh' not in st.session_state:
        st.session_state.refresh = False
    if 'department' not in st.session_state:
        st.session_state.department = None

def display_student_dashboard():
    st.title("Student Dashboard")
    
    if 'token' not in st.session_state or not st.session_state.token:
        st.error("No token found. Please log in again.")
        return

    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    logger.info(f"Token being sent: {st.session_state.token[:10]}...")
    
    try:
        response = requests.get(f"{API_URL}/student/stats", headers=headers)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.content}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Calculate attendance rate
            attendance_rate = data.get('attendance_rate', 0)
            
            # Show attendance alert if rate is below 30%
            if attendance_rate < 30:
                st.error(f"""
                    ⚠️ **Low Attendance Alert!**
                    
                    Your current attendance rate is {attendance_rate:.1f}%.
                    This is below the minimum required attendance of 30%.
                    Please improve your attendance to avoid academic penalties.
                    
                    Contact your class teacher for any clarifications.
                """)
            
            # Profile Section
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Display student's photo
                    face_image = data.get('face_image')
                    if face_image:
                        try:
                            # Decode and display the image
                            decoded_image = base64.b64decode(face_image)
                            nparr = np.frombuffer(decoded_image, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, width=150)
                        except Exception as e:
                            logger.error(f"Error displaying face image: {str(e)}")
                            st.image("https://ui-avatars.com/api/?name=Student&background=random", width=150)
                    else:
                        st.image("https://ui-avatars.com/api/?name=Student&background=random", width=150)
                
                with col2:
                    st.subheader("Your Overview")
                    
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        # Add color coding for attendance rate
                        rate_color = (
                            "#c62828" if attendance_rate < 30 else  # Dark red
                            "#e65100" if attendance_rate < 75 else  # Dark orange
                            "#33691e"  # Dark green
                        )
                        
                        st.markdown(
                            f"""
                            <div style='text-align: center; 
                                        padding: 1rem; 
                                        background-color: {'#ffebee' if attendance_rate < 30 else '#fff3e0' if attendance_rate < 75 else '#f1f8e9'}; 
                                        border-radius: 10px;
                                        margin: 1rem 0;'>
                                <h2 style='color: {rate_color}; margin: 0;'>
                                    {attendance_rate:.1f}%
                                </h2>
                                <p style='color: {rate_color}; margin: 0;'>Attendance Rate</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with metrics_cols[1]:
                        st.metric(
                            "Classes Attended",
                            data.get('present_classes', 'N/A'),
                            delta=data.get('classes_change', 0)
                        )
                    with metrics_cols[2]:
                        st.metric(
                            "Total Classes",
                            data.get('total_classes', 'N/A')
                        )

            # Attendance Records
            if "records" in data and data["records"]:
                df = pd.DataFrame(data["records"])
                st.subheader("Recent Attendance")
                
                # Add color coding to the dataframe
                def color_status(val):
                    colors = {
                        'present': 'background-color: #99ff99',
                        'absent': 'background-color: #ff9999',
                        'late': 'background-color: #ffcc99'
                    }
                    return colors.get(val, '')
                
                styled_df = df[['date', 'class_id', 'status']].style.applymap(
                    color_status, subset=['status']
                )
                st.dataframe(styled_df)
            else:
                st.info("No attendance records found")
                
            # Notifications Section
            with st.sidebar:
                st.subheader("📬 Notifications")
                if data.get("notifications"):
                    for notif in data["notifications"]:
                        with st.expander(f"📌 {notif.get('created_at', 'No date')}", expanded=False):
                            st.write(notif['message'])
                else:
                    st.write("No new notifications")
                
                # Add attendance status indicator in sidebar
                st.markdown("---")
                st.subheader("Attendance Status")
                if attendance_rate < 30:
                    st.markdown("""
                        <div style='padding: 1rem; 
                                  background-color: #ffebee; 
                                  border-left: 6px solid #ff1744;
                                  border-radius: 5px;
                                  margin: 1rem 0;'>
                            ⚠️ <b style='color: #c62828;'>Critical</b>: Attendance below minimum requirement
                        </div>
                    """, unsafe_allow_html=True)
                elif attendance_rate < 75:
                    st.markdown("""
                        <div style='padding: 1rem; 
                                  background-color: #fff3e0; 
                                  border-left: 6px solid #ff9100;
                                  border-radius: 5px;
                                  margin: 1rem 0;'>
                            ⚠️ <b style='color: #e65100;'>Warning</b>: Attendance below 75%
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='padding: 1rem; 
                                  background-color: #f1f8e9; 
                                  border-left: 6px solid #64dd17;
                                  border-radius: 5px;
                                  margin: 1rem 0;
                                  color: #33691e;'>
                            ✅ <b>Excellent</b>: Attendance above 75%
                        </div>
                    """, unsafe_allow_html=True)
                
        elif response.status_code == 401 or response.status_code == 403:
            logger.error("Authentication failed. Clearing session and redirecting to login.")
            st.error("Your session has expired or you're not authorized. Please log in again.")
            st.session_state.token = None
            st.session_state.role = None
            st.rerun()
        else:
            logger.error(f"Failed to fetch student statistics. Status code: {response.status_code}")
            logger.error(f"Error message: {response.text}")
            st.error("Unable to fetch dashboard data. Please try again later.")
    except Exception as e:
        logger.error(f"Error in display_student_dashboard: {str(e)}", exc_info=True)
        st.error("An error occurred while loading the dashboard. Please try again later.")



def display_admin_dashboard():
    st.title("Admin Dashboard")
    
    # System statistics
    st.subheader("System Overview")
    
    # Create metrics layout
    metrics_container = st.container()
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        
        headers = {'Authorization': f'Bearer {st.session_state.token}'}
        response = requests.get(f"{API_URL}/admin/stats", headers=headers)
        
        if response.status_code == 200:
            stats = response.json()
            
            with col1:
                st.metric(
                    "Total Users",
                    stats.get("total_users", "N/A"),
                    delta=stats.get("new_users_today", 0),
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Total Classes",
                    stats.get("total_classes", "N/A"),
                    delta=stats.get("new_classes_today", 0)
                )
            with col3:
                attendance_rate = stats.get("attendance_rate", 0)
                st.metric(
                    "Attendance Rate",
                    f"{attendance_rate}%",
                    delta=f"{stats.get('attendance_rate_change', 0)}%"
                )
            with col4:
                st.metric(
                    "Active Today",
                    stats.get("today_attendance", "N/A"),
                    delta=stats.get("attendance_change", 0)
                )

        # Activity Timeline
        st.subheader("Recent Activity")
        activity_container = st.container()
        with activity_container:
            if stats.get("recent_activity"):
                for activity in stats["recent_activity"]:
                    with st.expander(f"{activity['timestamp']} - {activity['type']}", expanded=False):
                        st.write(activity['description'])
                        if activity.get('details'):
                            st.json(activity['details'])
            else:
                st.info("No recent activity to display")

        # User Distribution
        st.subheader("User Distribution")
        charts_container = st.container()
        with charts_container:
            col1, col2 = st.columns(2)
            
            with col1:
                user_stats = pd.DataFrame(stats["user_distribution"].items(), columns=['Role', 'Count'])
                fig = px.pie(
                    user_stats,
                    values='Count',
                    names='Role',
                    title='Users by Role',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if stats.get("class_distribution"):
                    class_stats = pd.DataFrame(stats["class_distribution"].items(), columns=['Department', 'Count'])
                    fig = px.bar(
                        class_stats,
                        x='Department',
                        y='Count',
                        title='Classes by Department',
                        color='Department',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Attendance Trends
        st.subheader("Attendance Trends")
        if stats.get("attendance_trends"):
            trends_df = pd.DataFrame(stats["attendance_trends"])
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trends_df['date'],
                y=trends_df['attendance_rate'],
                mode='lines+markers',
                name='Attendance Rate',
                line=dict(color='#2E86C1', width=2)
            ))
            
            fig.update_layout(
                title='Daily Attendance Rate',
                xaxis_title='Date',
                yaxis_title='Attendance Rate (%)',
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to load admin statistics")


def mark_attendance_from_photo(class_photo, known_face_encodings, known_face_ids, known_face_images, tolerance=0.6):
    # Convert class photo to numpy array
    image = face_recognition.load_image_file(class_photo)
    
    # Detect faces in the photo
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    logger.info(f"Detected {len(face_encodings)} faces in the uploaded photo")
    logger.info(f"Have {len(known_face_encodings)} known faces to compare against")

    marked_students = {}
    
    for i, face_encoding in enumerate(face_encodings):
        if len(known_face_encodings) > 0:  # Only try matching if we have known faces
            # Calculate face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            logger.info(f"Face {i}: Best match distance = {min_distance}")
            
            if min_distance < tolerance:
                student_id = known_face_ids[best_match_index]
                if student_id not in marked_students:
                    marked_students[student_id] = known_face_images[best_match_index]
                    logger.info(f"Matched student ID: {student_id} with confidence: {1 - min_distance:.2f}")
            else:
                logger.warning(f"Face {i}: No match found (min_distance = {min_distance})")
        else:
            logger.warning("No known faces available for comparison")

    return marked_students

def update_attendance_records(class_id, marked_students):
    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    successful_marks = []
    
    current_date = datetime.now().strftime('%Y-%m-%d')  # Get current date
    
    for student_id, face_bytes in marked_students.items():
        logger.info(f"Sending attendance update for student ID: {student_id} in class {class_id}")
        
        # Create form data with the cropped face image
        files = {'face_image': ('face.jpg', face_bytes, 'image/jpeg')}
        form_data = {
            'class_id': class_id,
            'student_id': student_id,
            'date': current_date,  # Add date field
            'status': 'present'
        }
        
        try:
            # Log the request data for debugging
            logger.info(f"Sending request with data: {form_data}")
            
            response = requests.post(
                f"{API_URL}/attendance/mark",
                files=files,
                data=form_data,
                headers=headers
            )
            
            logger.info(f"API response status code: {response.status_code}")
            logger.info(f"API response content: {response.content}")
            
            if response.status_code == 200:
                logger.info(f"Attendance record updated for student ID: {student_id}")
                successful_marks.append(student_id)
            else:
                error_msg = response.json().get('detail', 'Unknown error') if response.content else 'No response content'
                logger.error(f"Failed to update attendance for student ID: {student_id}. Error: {error_msg}")
                logger.error(f"Full response: {response.text}")
                st.error(f"Failed to update attendance for student ID: {student_id}. Error: {error_msg}")
        except Exception as e:
            logger.error(f"Exception while updating attendance for student ID: {student_id}. Error: {str(e)}")
            st.error(f"Error updating attendance for student ID: {student_id}. Please try again.")
    
    return successful_marks

def display_teacher_dashboard():
    st.title("Teacher Dashboard")
    
    # Add debug button
    if st.button("Debug Face Encodings"):
        debug_face_encodings()
    
    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    
    with st.expander("Quick Actions", expanded=True):
        class_id = st.text_input("Class ID")
        class_photo = st.file_uploader("Upload Class Photo", type=['jpg', 'jpeg', 'png'])
        
        if st.button("Mark Attendance from Class Photo"):
            if not class_id or not class_photo:
                st.error("Please provide a class ID and upload a class photo.")
                return
            
            known_face_encodings, known_face_ids, known_face_images = load_known_faces()
            
            if not known_face_encodings:
                st.error("No registered faces found in the database. Please ensure students are registered with their photos.")
                return
                
            with st.spinner('Processing class photo...'):
                marked_students = mark_attendance_from_photo(
                    class_photo, 
                    known_face_encodings, 
                    known_face_ids,
                    known_face_images,
                    tolerance=0.6
                )
            
            if marked_students:
                successful_marks = update_attendance_records(class_id, marked_students)
                if successful_marks:
                    st.success(f"Attendance marked for students: {', '.join(successful_marks)}")
                    st.session_state.refresh = True
                else:
                    st.warning("No attendance records were successfully updated.")
            else:
                st.error("Failed to mark attendance: No known faces detected")

    # Class Selection with search
    department = st.session_state.get('department', 'Unknown')
    class_options = ["C123", "C456", "C789"] if department == "Computer Science" else ["E123", "E456", "E789"]
    class_id = st.selectbox("Select Class", class_options)
    
    # Date Range Selection
    date_cols = st.columns(2)
    with date_cols[0]:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=7),
            max_value=datetime.now()
        )
    with date_cols[1]:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            max_value=datetime.now()
        )
    
    # Fetch attendance data
    logger.info(f"Fetching attendance data for class {class_id} from {start_date} to {end_date}")
    response = requests.get(
        f"{API_URL}/attendance/class/{class_id}",
        params={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        headers=headers
    )
    
    logger.info(f"API response status code: {response.status_code}")
    logger.info(f"API response content: {response.content}")

    if response.status_code == 200:
        data = response.json()
        logger.info(f"Received data: {data}")
        
        if data["records"]:
            df = pd.DataFrame(data["records"])
            df['date'] = pd.to_datetime(df['date']).dt.date
            logger.info(f"Created DataFrame with {len(df)} rows")
            
            # Display summary of attendance records
            st.subheader("Attendance Summary")
            st.dataframe(
                df[['student_id', 'name', 'department', 'class_id', 'date', 'status']],
                column_config={
                    "student_id": "Student ID",
                    "name": "Name",
                    "department": "Department",
                    "class_id": "Class ID",
                    "date": "Date",
                    "status": "Status"
                },
                hide_index=True
            )
            
            # Statistics Cards
            stat_cols = st.columns(2)
            with stat_cols[0]:
                present_rate = len(df[df['status'] == 'present']) / len(df) * 100
                st.metric("Attendance Rate", f"{present_rate:.1f}%")
            with stat_cols[1]:
                st.metric("Total Students", len(df['student_id'].unique()))
            
            # Attendance Visualization
            st.subheader("Attendance Overview")
            viz_cols = st.columns(2)
            
            with viz_cols[0]:
                # Daily attendance trend
                daily_attendance = df.groupby('date')['status'].value_counts().unstack().fillna(0)
                fig = px.bar(
                    daily_attendance,
                    barmode='group',
                    title='Daily Attendance',
                    color_discrete_map={'present': '#99ff99', 'absent': '#ff9999', 'late': '#ffcc99'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_cols[1]:
                # Overall attendance distribution
                overall_attendance = df['status'].value_counts()
                fig = px.pie(
                    names=overall_attendance.index,
                    values=overall_attendance.values,
                    title='Overall Attendance Distribution',
                    color_discrete_map={'present': '#99ff99', 'absent': '#ff9999', 'late': '#ffcc99'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual Student Attendance Records with inline editing
            st.subheader("Individual Student Attendance Records")
            for student_id, student_group in df.groupby('student_id'):
                with st.expander(f"Student {student_id} - {student_group['name'].iloc[0]}", expanded=False):
                    student_df = student_group.sort_values('date', ascending=False)
                    edited_df = st.data_editor(
                        student_df[['date', 'status']],
                        column_config={
                            "date": st.column_config.DateColumn("Date", disabled=True),
                            "status": st.column_config.SelectboxColumn(
                                "Status",
                                options=["present", "absent", "late"],
                                required=True
                            )
                        },
                        hide_index=True,
                        key=f"editor_{student_id}"
                    )
                    
                    # Check for changes and update
                    if not edited_df.equals(student_df[['date', 'status']]):
                        changes = edited_df[edited_df['status'] != student_df['status']]
                        for index, row in changes.iterrows():
                            update_response = requests.put(
                                f"{API_URL}/attendance/update",
                                json={
                                    "student_id": student_id,
                                    "class_id": class_id,
                                    "date": row['date'].isoformat(),
                                    "status": row['status']
                                },
                                headers=headers
                            )
                            if update_response.status_code == 200:
                                st.success(f"Updated attendance for {student_id} on {row['date']}")
                            else:
                                st.error(f"Failed to update attendance for {student_id} on {row['date']}")
                        
                        # Refresh the data after updates
                        st.rerun()
        
        else:
            st.info("No attendance records found for the selected period")
            logger.warning(f"No records found for class {class_id} from {start_date} to {end_date}")
    else:
        st.error("Failed to fetch attendance data")
        logger.error(f"Failed to fetch attendance data. Status code: {response.status_code}, Content: {response.content}")


def sidebar_menu():
    st.sidebar.title("Navigation")
    menu_options = ["Dashboard", "Mark Attendance", "Edit Attendance", "Reports"]
    return st.sidebar.radio("Select a view:", menu_options)


def display_dashboard():
    st.sidebar.title(f"Welcome, {st.session_state.role.capitalize()}")
    
    if st.sidebar.button("Logout"):
        for key in ['token', 'role', 'auth_type', 'current_view', 'page']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "landing"
        st.rerun()
    
    # Restrict access based on role
    if st.session_state.role == 'student':
        display_student_dashboard()
    else:
        view = sidebar_menu()
        
        if view == 'Dashboard':
            if st.session_state.role == 'teacher':
                display_teacher_dashboard()
            elif st.session_state.role == 'admin':
                display_admin_dashboard()
        elif view == 'Mark Attendance':
            mark_attendance()
        elif view == 'Edit Attendance':
            edit_attendance()
        elif view == 'Reports':
            generate_reports()
    
    # Add debug section
    if st.sidebar.checkbox("Debug Token"):
        debug_token()

def clear_face_encodings():
    """Clear the face encodings file and start fresh"""
    try:
        with open('face_encodings.json', 'w') as file:
            json.dump({}, file)
        logger.info("Face encodings file cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing face encodings: {str(e)}")
        return False

def register_student():
    st.title("Student Registration")
    
    department = st.session_state.get('department', 'Unknown')
    classes = ["E123", "E456", "E789"] if department == "Electronics and Communication" else ["C123", "C456", "C789"]
    
    with st.form("student_registration_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = st.text_input("Full Name")
        student_id = st.text_input("Student ID")
        class_id = st.selectbox("Class", classes)
        face_image = st.file_uploader("Upload your face photo", type=['jpg', 'png'])
        
        submit = st.form_submit_button("Register")
        
        if submit and email and password and name and student_id and class_id and face_image:
            try:
                # Process face image and create form data
                files = {'face_image': ('face.jpg', face_image.getvalue(), 'image/jpeg')}
                data = {
                    'email': email,
                    'password': password,
                    'name': name,
                    'student_id': student_id,
                    'class_id': class_id,
                    'department': department  # Include department in registration data
                }
                
                response = requests.post(f"{API_URL}/register/student", files=files, data=data)
                
                if response.status_code == 200:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error during registration: {str(e)}")
                st.error("An error occurred during registration. Please try again.")

def admin_login():
    st.title("Admin Login")
    
    with st.form("admin_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login as Admin")
        
        if submit and email and password:
            response = requests.post(
                f"{API_URL}/admin/login",
                data={"username": email, "password": password}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                st.session_state.token = token_data["access_token"]
                st.session_state.role = "admin"
                st.success("Admin login successful!")
                st.rerun()
            else:
                st.error("Admin login failed. Please check your credentials.")

def mark_attendance():
    st.title("Mark Attendance")
    
    with st.form("attendance_form"):
        class_id = st.text_input("Class ID")
        student_id = st.text_input("Student ID")
        date = st.date_input("Date", datetime.now())
        photo = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
        
        submit = st.form_submit_button("Mark Attendance")
        
        if submit:
            if not all([class_id, student_id, photo]):
                st.error("Please fill all required fields and upload a photo.")
                return
            
            try:
                # Create form data
                files = {
                    'face_image': (photo.name, photo.getvalue(), 'image/jpeg')
                }
                
                form_data = {
                    'class_id': class_id,
                    'student_id': student_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'status': 'present'
                }
                
                headers = {
                    'Authorization': f'Bearer {st.session_state.token}'
                }
                
                with st.spinner('Processing attendance...'):
                    response = requests.post(
                        f"{API_URL}/attendance/mark",
                        files=files,
                        data=form_data,
                        headers=headers
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Attendance marked successfully! Similarity score: {result.get('similarity_score', 'N/A')}")
                else:
                    error_msg = response.json().get('detail', 'Unknown error occurred')
                    st.error(f"Failed to mark attendance: {error_msg}")
                    logger.error(f"Failed to mark attendance. Response: {response.json()}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error marking attendance: {str(e)}")

def edit_attendance():
    st.title("Edit Attendance")
    
    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    
    # Class Selection
    department = st.session_state.get('department', 'Unknown')
    class_options = ["C123", "C456", "C789"] if department == "Computer Science" else ["E123", "E456", "E789"]
    class_id = st.selectbox("Select Class", class_options)
    
    # Date Selection
    date = st.date_input("Select Date", datetime.now())
    
    # Store attendance data in session state
    if 'attendance_data' not in st.session_state:
        st.session_state.attendance_data = None
    
    if st.button("Fetch Attendance") or st.session_state.attendance_data is None:
        response = requests.get(
            f"{API_URL}/attendance/class/{class_id}",
            params={"start_date": date.isoformat()},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["records"]:
                st.session_state.attendance_data = data["records"]
            else:
                st.session_state.attendance_data = None
                st.info("No attendance records found for the selected date")
        else:
            st.error("Failed to fetch attendance data")
    
    # Display and edit attendance records
    if st.session_state.attendance_data:
        for record in st.session_state.attendance_data:
            student_id = record['student_id']
            name = record['name']
            current_status = record['status']
            
            # Create columns for layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{name}** (ID: {student_id})")
            
            with col2:
                new_status = st.selectbox(
                    "Status",
                    options=["present", "absent", "late"],
                    index=["present", "absent", "late"].index(current_status),
                    key=f"status_{student_id}"
                )
            
            with col3:
                if new_status != current_status:
                    if st.button("Update", key=f"btn_{student_id}"):
                        # Create form data
                        form_data = {
                            "student_id": student_id,
                            "class_id": class_id,
                            "date": date.isoformat(),
                            "status": new_status
                        }
                        
                        response = requests.post(  # Changed to POST
                            f"{API_URL}/attendance/update_status",  # New endpoint
                            data=form_data,  # Send as form data
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            # Update the record in session state
                            for record in st.session_state.attendance_data:
                                if record['student_id'] == student_id:
                                    record['status'] = new_status
                            st.success(f"Updated attendance for {name}")
                        else:
                            st.error(f"Failed to update attendance: {response.text}")
        
        # Add a refresh button at the bottom
        if st.button("Refresh Data"):
            st.session_state.attendance_data = None
            st.rerun()

def generate_reports():
    st.title("Generate Attendance Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Student-wise", "Class-wise", "Overall Statistics"]
    )
    
    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    
    if report_type == "Student-wise":
        student_id = st.text_input("Enter Student ID")
        if student_id:
            try:
                response = requests.get(
                    f"{API_URL}/attendance/student/{student_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, dict):
                        # Display student information
                        st.subheader("Student Information")
                        student_info = data.get("student_info", {})
                        
                        # Create metrics for student info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Name", student_info.get("name", "Unknown"))
                        with col2:
                            department = student_info.get("department", "Unknown")
                            st.metric("Department", department)
                        with col3:
                            class_id = student_info.get("class_id", "Unknown")
                            st.metric("Class", class_id)
                        
                        # Display statistics
                        st.subheader("Attendance Statistics")
                        stats = data.get("statistics", {})
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            total_classes = stats.get("total_classes", 0)
                            present_count = stats.get("present", 0)
                            attendance_rate = (present_count / total_classes * 100) if total_classes > 0 else 0
                            st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
                        with stat_cols[1]:
                            st.metric("Present", stats.get("present", 0))
                        with stat_cols[2]:
                            st.metric("Absent", stats.get("absent", 0))
                        with stat_cols[3]:
                            st.metric("Late", stats.get("late", 0))
                        
                        # Display attendance records
                        records = data.get("records", [])
                        if records:
                            st.subheader("Attendance Records")
                            df = pd.DataFrame(records)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.sort_values('date')
                            
                            # Display records in a table
                            st.dataframe(
                                df[['date', 'class_id', 'status']].style.format({
                                    'date': lambda x: x.strftime('%Y-%m-%d')
                                })
                            )
                            
                            # Attendance trend visualization
                            st.subheader("Attendance Trend")
                            
                            # Convert status to numeric values for visualization
                            status_map = {'present': 1, 'late': 0.5, 'absent': 0}
                            df['status_numeric'] = df['status'].map(status_map)
                            
                            # Create the trend line chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['status_numeric'],
                                mode='lines+markers',
                                name='Attendance Status',
                                line=dict(color='#2E86C1', width=2),
                                marker=dict(
                                    size=8,
                                    color=df['status_numeric'].map({1: '#28a745', 0.5: '#ffc107', 0: '#dc3545'}),
                                    symbol='circle'
                                )
                            ))
                            
                            fig.update_layout(
                                title='Attendance Status Over Time',
                                xaxis_title='Date',
                                yaxis=dict(
                                    title='Status',
                                    ticktext=['Absent', 'Late', 'Present'],
                                    tickvals=[0, 0.5, 1],
                                    range=[-0.1, 1.1]
                                ),
                                hovermode='x unified',
                                showlegend=True
                            )
                            
                            # Add hover template
                            fig.update_traces(
                                hovertemplate="<br>".join([
                                    "Date: %{x|%Y-%m-%d}",
                                    "Status: %{text}",
                                ]),
                                text=df['status']
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add monthly attendance rate chart
                            st.subheader("Monthly Attendance Rate")
                            monthly_stats = df.set_index('date').resample('M').agg({
                                'status_numeric': 'mean'
                            }).reset_index()
                            
                            monthly_fig = go.Figure()
                            monthly_fig.add_trace(go.Bar(
                                x=monthly_stats['date'],
                                y=monthly_stats['status_numeric'] * 100,
                                name='Monthly Rate',
                                marker_color='#3498db'
                            ))
                            
                            monthly_fig.update_layout(
                                title='Monthly Attendance Rate',
                                xaxis_title='Month',
                                yaxis_title='Attendance Rate (%)',
                                yaxis_range=[0, 100],
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(monthly_fig, use_container_width=True)
                            
                            # Download report option
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Attendance Report",
                                csv,
                                f"attendance_report_{student_id}.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.info("No attendance records found for this student")
                    else:
                        st.error("Invalid data format received from server")
                else:
                    st.error(f"Failed to fetch student data: {response.text}")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    elif report_type == "Class-wise":
        class_id = st.selectbox("Select Class", ["E123", "E456", "E789", "C123", "C456", "C789"])
        response = requests.get(
            f"{API_URL}/attendance/stats/{class_id}",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            
            # Display statistics
            st.subheader("Class Statistics")
            stats_df = pd.DataFrame(data["overall_stats"].items(), columns=['Status', 'Count'])
            fig = px.bar(stats_df, x='Status', y='Count', title='Attendance Distribution')
            st.plotly_chart(fig)
            
            # Student-wise breakdown
            st.subheader("Student-wise Statistics")
            student_stats = pd.DataFrame(data["student_stats"]).T
            student_stats['Attendance Rate'] = (student_stats['present'] / student_stats['total']) * 100
            st.dataframe(student_stats)
            
            # Download report
            csv_data = student_stats.to_csv(index=True)
            st.download_button(
                "Download Detailed Class Report",
                csv_data,
                f"class_{class_id}_attendance_report.csv",
                "text/csv"
            )
        else:
            st.error("Failed to fetch class attendance statistics.")
    
    elif report_type == "Overall Statistics":
        response = requests.get(
            f"{API_URL}/attendance/overall_stats",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            st.subheader("Overall Attendance Statistics")
            
            # Display overall statistics
            overall_stats_df = pd.DataFrame(data["overall_stats"].items(), columns=['Status', 'Count'])
            fig = px.bar(overall_stats_df, x='Status', y='Count', title='Overall Attendance Distribution')
            st.plotly_chart(fig)
            
            # Calculate and display attendance rates
            total_attendance = overall_stats_df['Count'].sum()
            overall_stats_df['Rate (%)'] = (overall_stats_df['Count'] / total_attendance) * 100
            st.dataframe(overall_stats_df)
            
            # Download overall statistics report
            csv_data = overall_stats_df.to_csv(index=False)
            st.download_button(
                "Download Overall Statistics Report",
                csv_data,
                "overall_attendance_report.csv",
                "text/csv"
            )
        else:
            st.error("Failed to fetch overall attendance statistics.")

def login(email, password):
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": email,
                "password": password,
                "department": st.session_state.department
            }
        )
        logger.info(f"Login response status code: {response.status_code}")
        logger.info(f"Login response content: {response.content}")
        
        if response.status_code == 200:
            try:
                token_data = response.json()
                st.session_state.token = token_data["access_token"]
                st.session_state.role = token_data["role"]
                st.session_state.user_department = token_data["department"]
                logger.info(f"Login successful for user: {email}")
                logger.info(f"Token stored: {st.session_state.token[:10]}...")
                logger.info(f"Role: {st.session_state.role}")
                logger.info(f"Department: {st.session_state.user_department}")
                return True
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to decode JSON response: {str(json_error)}")
                st.error("An error occurred while processing the server response. Please try again.")
                return False
        else:
            try:
                error_message = response.json().get('detail', 'Unknown error')
            except json.JSONDecodeError:
                error_message = response.text if response.text else "Unknown error"
            logger.warning(f"Login failed for user {email}: {error_message}")
            st.error(f"Login failed: {error_message}")
            return False
    except requests.RequestException as e:
        logger.error(f"Request error during login: {str(e)}", exc_info=True)
        st.error(f"An error occurred while connecting to the server: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred during login: {str(e)}")
        return False

def get_student_attendance(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/students/attendance", headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def admin_login():
    st.title("Admin Login")
    
    with st.form("admin_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login as Admin")
        
        if submit and email and password:
            response = requests.post(
                f"{API_URL}/admin/login",
                data={"username": email, "password": password}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                st.session_state.token = token_data["access_token"]
                st.session_state.role = "admin"
                st.success("Admin login successful!")
                st.rerun()
            else:
                st.error("Admin login failed. Please check your credentials.")

def register_teacher(name, email, password):
    try:
        response = requests.post(f"{API_URL}/register/teacher", json={"name": name, "email": email, "password": password})
        if response.status_code == 200:
            logger.info(f"Teacher registered successfully: {email}")
            st.success("Teacher registered successfully!")
        else:
            error_message = response.json().get('detail', 'Unknown error')
            logger.warning(f"Teacher registration failed for {email}: {error_message}")
            st.error(f"Teacher registration failed: {error_message}")
    except Exception as e:
        logger.error(f"Error during teacher registration: {str(e)}")
        st.error(f"An error occurred during registration: {str(e)}")

def register_teacher():
    st.title("Teacher Registration")
    
    with st.form("teacher_registration_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit and name and email and password:
            response = requests.post(
                f"{API_URL}/register/teacher",
                json={"name": name, "email": email, "password": password}
            )
            
            if response.status_code == 200:
                st.success("Teacher registration successful! Please login.")
            else:
                st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")

def student_registration():
    st.title("Student Registration")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    name = st.text_input("Name")
    student_id = st.text_input("Student ID")
    face_image = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'])

    if st.button("Register"):
        if not all([email, password, name, student_id, face_image]):
            st.error("Please fill all fields and upload a face image.")
            return

        files = {'face_image': (face_image.name, face_image.getvalue(), 'image/jpeg')}
        form_data = {
            'email': email,
            'password': password,
            'name': name,
            'student_id': student_id
        }

        response = requests.post(f"{API_URL}/register/student", files=files, data=form_data)
        if response.status_code == 200:
            st.success("Registration successful!")
        else:
            st.error("Registration failed.")

def review_and_edit_attendance():
    st.title("Review and Edit Attendance")
    
    class_id = st.text_input("Class ID")
    date = st.date_input("Date", datetime.now())
    
    if st.button("Fetch Attendance"):
        headers = {'Authorization': f'Bearer {st.session_state.token}'}
        response = requests.get(f"{API_URL}/attendance/review", params={'class_id': class_id, 'date': date}, headers=headers)
        
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                df = pd.DataFrame(records)
                st.dataframe(df)
                
                # Allow editing
                for index, row in df.iterrows():
                    new_status = st.selectbox(f"Status for {row['student_id']}", ['present', 'absent'], index=['present', 'absent'].index(row['status']))
                    if new_status != row['status']:
                        response = requests.post(f"{API_URL}/attendance/edit", json={'attendance_id': row['id'], 'status': new_status}, headers=headers)
                        if response.status_code == 200:
                            st.success(f"Updated status for {row['student_id']}")
                        else:
                            st.error(f"Failed to update status for {row['student_id']}")
            else:
                st.info("No records found for the given class and date.")
        else:
            st.error("Failed to fetch attendance records.")

def display_notifications():
    st.sidebar.title("Notifications")
    headers = {'Authorization': f'Bearer {st.session_state.token}'}
    response = requests.get(f"{API_URL}/notifications", headers=headers)
    
    if response.status_code == 200:
        notifications = response.json().get('notifications', [])
        if notifications:
            for notif in notifications:
                st.sidebar.write(f"{notif['date']}: {notif['message']}")
        else:
            st.sidebar.info("No new notifications.")
    else:
        st.sidebar.error("Failed to fetch notifications.")

def debug_token():
    if 'token' in st.session_state and st.session_state.token:
        headers = {'Authorization': f'Bearer {st.session_state.token}'}
        response = requests.get(f"{API_URL}/debug/token", headers=headers)
        st.json(response.json())
    else:
        st.error("No token found in session state")

def debug_face_encodings():
    try:
        with open('face_encodings.json', 'r') as file:
            data = json.load(file)
            logger.info("Current face encodings file content structure:")
            for student_id, student_data in data.items():
                logger.info(f"Student ID: {student_id}")
                if isinstance(student_data, dict):
                    logger.info(f"  Has encoding: {'encoding' in student_data}")
                    logger.info(f"  Has image: {'image' in student_data}")
                    if 'encoding' in student_data:
                        logger.info(f"  Encoding length: {len(student_data['encoding'])}")
                else:
                    logger.info(f"  Invalid data type: {type(student_data)}")
            
            # Display the raw data for debugging
            logger.info("Raw data structure:")
            logger.info(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Error reading face encodings file: {str(e)}")

def main():
    init_session_state()
    
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "department_select":
        select_department()
    elif st.session_state.page == "login_register":
        if st.session_state.department:
            display_login_register()
        else:
            st.session_state.page = "department_select"
            st.rerun()
    else:
        display_dashboard()

def select_department():
    st.title("Select Department")
    department = st.selectbox("Choose your department", ["Electronics and Communication", "Computer Science"])
    if st.button("Continue"):
        st.session_state.department = department
        st.session_state.page = "login_register"
        st.rerun()

def display_login_register():
    st.title("AttendAI")
    
    department = st.session_state.get('department', 'Unknown')
    st.write(f"Department: {department}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Student Login", "Teacher Login", "Student Registration", "Teacher Registration"])
    
    with tab1:
        with st.form("student_login_form"):
            email = st.text_input("Student Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login as Student")
            
            if submit and email and password:
                if login(email, password):
                    st.session_state.page = "dashboard"
                    st.rerun()

    with tab2:
        with st.form("teacher_login_form"):
            email = st.text_input("Teacher Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login as Teacher")
            
            if submit and email and password:
                if login(email, password):
                    st.session_state.page = "dashboard"
                    st.rerun()
    
    with tab3:
        register_student()
    
    with tab4:
        register_teacher()

if __name__ == "__main__":
    main()

# Check if we need to refresh the page
if st.session_state.get('refresh', False):
    st.session_state.refresh = False
    # Use JavaScript to reload the page
    st.write('<script>location.reload()</script>', unsafe_allow_html=True)








