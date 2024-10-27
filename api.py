from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, date
import cv2
import numpy as np
import json
import base64
import sqlite3
import jwt
from jwt.exceptions import PyJWTError
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
import logging
import uuid
from scipy.spatial.distance import cosine
from jose import JWTError
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from database import (
    init_db,
    add_user, 
    record_attendance, 
    add_notification,
    get_user_by_email,
    get_class_attendance,
    update_attendance_record,
    get_student_attendance,
    get_class_stats,
    get_class_attendance_range,
    get_user_notifications,  # Add this line
    migrate_db,  # Add this line
    get_db,  # Add this line
    update_user_face_encoding,  # Add this line
    get_user_by_student_id,  # Add this line
    migrate_notifications_table,  # Add this line
    fetch_attendance_records,  # Add this line
    mark_absent_students  # Add this line
)
from auth import (
    get_password_hash,
    create_access_token,
    authenticate_user,
    decode_token
)

app = FastAPI()
init_db()
migrate_db()  # Ensure this is called to update the schema

# Load environment variables
load_dotenv()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserRegistration(BaseModel):
    email: str
    password: str
    name: str
    student_id: str
    face_encoding: str

class AttendanceRecord(BaseModel):
    student_id: str
    class_id: str
    date: date
    status: str

class AttendanceUpdate(BaseModel):
    status: str

class DateRange(BaseModel):
    start_date: date
    end_date: date

class TeacherRegistration(BaseModel):
    name: str
    email: str
    password: str

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))  # Default to 30 minutes if not set

def is_face_match(encoding1, encoding2, threshold=0.4):
    """Compare two face encodings and return True if they match."""
    return cosine(encoding1, encoding2) < threshold

@app.post("/register")
async def register_user(user: UserRegistration):
    try:
        # Check if email is already registered
        existing_user = get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        password_hash = get_password_hash(user.password)
        user_id = add_user(user.email, password_hash, "student", user.name, user.student_id, user.face_encoding)
        
        add_notification(user_id, f"Welcome to AttendAI! Your student account has been created successfully.")
        
        return {"message": "Registration successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/register/teacher")
async def register_teacher(teacher: TeacherRegistration):
    try:
        logger.info(f"Attempting to register teacher: {teacher.email}")
        # Check if email is already registered
        existing_user = get_user_by_email(teacher.email)
        if existing_user:
            logger.warning(f"Email already registered: {teacher.email}")
            raise HTTPException(status_code=400, detail="Email already registered")

        password_hash = get_password_hash(teacher.password)
        
        # Add user to database
        try:
            user_id = add_user(teacher.email, password_hash, "teacher", teacher.name, None, None)
            
            # Try to add notification, but don't fail if it doesn't work
            try:
                add_notification(user_id, f"Welcome to AttendAI! Your teacher account has been created successfully.")
            except Exception as e:
                logger.warning(f"Failed to add welcome notification: {str(e)}")
            
            logger.info(f"Teacher registered successfully: {teacher.email}")
            return {"message": "Teacher registration successful"}
            
        except Exception as e:
            logger.error(f"Error adding teacher to database: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to register teacher")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error registering teacher: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Update the Pydantic model
class AttendanceMarkRequest(BaseModel):
    class_id: str
    student_id: str
    date: Optional[str] = None
    status: str = "present"  # Set a default value

@app.post("/attendance/mark")
async def mark_attendance(
    class_id: str = Form(...),
    student_id: str = Form(...),
    date: Optional[str] = Form(None),
    status: str = Form("present"),
    face_image: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    # Validate token and get payload
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get student data
    student = get_user_by_student_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Process face encoding
    face_encoding = student.get("face_encoding")
    if not face_encoding:
        raise HTTPException(status_code=400, detail="No face encoding registered for this student")

    contents = await face_image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Save image temporarily
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Process image with DeepFace
        face_encodings = DeepFace.represent(
            img_path=temp_path,
            model_name="VGG-Face",
            enforce_detection=True,
            detector_backend="retinaface"
        )
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No faces detected in the image")

        # Compare with registered face encoding
        input_face_encoding = face_encodings[0]['embedding']
        input_face_encoding_b64 = base64.b64encode(np.array(input_face_encoding).tobytes()).decode('utf-8')

        if input_face_encoding_b64 != face_encoding:
            raise HTTPException(status_code=400, detail="Face does not match registered student")

        # Record attendance
        attendance_date = date if date else datetime.now().date()
        record_attendance(student_id, class_id, attendance_date, status)
        
        return {"message": "Attendance marked successfully"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/attendance/class/{class_id}")
async def get_class_attendance(
    class_id: str,
    start_date: str,
    end_date: Optional[str] = None,
    token: str = Depends(oauth2_scheme)
):
    logger.info(f"Fetching attendance for class {class_id} from {start_date} to {end_date or start_date}")
    try:
        # Verify token and user role
        payload = decode_token(token)
        if not payload or payload.get("role") not in ["teacher", "admin"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        # If end_date is not provided, use start_date for both
        end_date = end_date or start_date

        # Fetch attendance records
        records = fetch_attendance_records(class_id, start_date, end_date)
        logger.info(f"Found {len(records)} attendance records")
        return {"records": records}
    except Exception as e:
        logger.error(f"Error fetching attendance records: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Update the database function to return serializable objects
def get_class_attendance(class_id: str, date: date):
    try:
        # Example implementation - adjust according to your database schema
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, student_id, date, status 
            FROM attendance 
            WHERE class_id = ? AND date = ?
        """, (class_id, date.isoformat()))
        
        records = cursor.fetchall()
        conn.close()
        
        # Convert to dictionary format
        return [
            {
                "id": record[0],
                "student_id": record[1],
                "date": record[2],
                "status": record[3]
            }
            for record in records
        ]
    except Exception as e:
        logger.error(f"Database error in get_class_attendance: {str(e)}")
        raise Exception(f"Database error: {str(e)}")

@app.put("/attendance/{record_id}")
async def update_attendance(
    record_id: int,
    update: AttendanceUpdate,
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="Not authorized")

        update_attendance_record(record_id, update.status)
        return {"message": "Attendance updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/attendance/student/{student_id}")
async def get_student_attendance_report(
    student_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or (payload.get("role") not in ["teacher", "admin"] and payload.get("sub") != student_id):
            raise HTTPException(status_code=403, detail="Not authorized")

        logger.info(f"Fetching attendance records for student {student_id}")
        
        # Get student info and attendance records
        with get_db() as conn:
            c = conn.cursor()
            
            # Get student details with proper department determination
            c.execute("""
                SELECT name, class_id, department
                FROM users
                WHERE student_id = ?
            """, (student_id,))
            student = c.fetchone()
            
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")
            
            name, class_id, stored_department = student
            
            # Determine department based on class_id if not stored
            if not stored_department:
                if class_id and class_id.startswith('E'):
                    department = "Electronics and Communication"
                elif class_id and class_id.startswith('C'):
                    department = "Computer Science"
                else:
                    department = "Unknown"
            else:
                department = stored_department
            
            # Get attendance records
            c.execute("""
                SELECT a.date, a.class_id, a.status
                FROM attendance a
                WHERE a.student_id = ?
                ORDER BY a.date DESC
            """, (student_id,))
            records = c.fetchall()
            
            response = {
                "student_info": {
                    "name": name or "Unknown",
                    "department": department,
                    "class_id": class_id or "Unknown"
                },
                "records": [
                    {
                        "date": str(record[0]),
                        "class_id": record[1],
                        "status": record[2]
                    }
                    for record in records
                ],
                "statistics": {
                    "total_classes": len(records),
                    "present": len([r for r in records if r[2] == 'present']),
                    "absent": len([r for r in records if r[2] == 'absent']),
                    "late": len([r for r in records if r[2] == 'late'])
                }
            }
            
            logger.info(f"Successfully retrieved attendance records: {response}")
            return response

    except Exception as e:
        logger.error(f"Error fetching student attendance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def get_student_info(student_id: str):
    """Get student information."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT name, class_id
                FROM users
                WHERE student_id = ?
            """, (student_id,))
            student = c.fetchone()
            
            if student:
                class_id = student[1] if student[1] else "Unknown"
                department = "Electronics and Communication" if class_id.startswith('E') else "Computer Science" if class_id.startswith('C') else "Unknown"
                return {
                    "name": student[0] or "Unknown",
                    "department": department,
                    "class_id": class_id
                }
            return {
                "name": "Unknown",
                "department": "Unknown",
                "class_id": "Unknown"
            }
    except Exception as e:
        logger.error(f"Error getting student info: {str(e)}", exc_info=True)
        return {
            "name": "Unknown",
            "department": "Unknown",
            "class_id": "Unknown"
        }

@app.get("/attendance/stats/{class_id}")
async def get_class_attendance_stats(
    class_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") not in ["teacher", "admin"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        stats = get_class_stats(class_id)  # Ensure this returns serializable data
        
        # Ensure the stats are serializable
        serializable_stats = {
            "overall_stats": {
                "present": stats.get("overall_stats", {}).get("present", 0),
                "absent": stats.get("overall_stats", {}).get("absent", 0),
                "late": stats.get("overall_stats", {}).get("late", 0)
            },
            "student_stats": {
                student_id: {
                    "present": data.get("present", 0),
                    "total": data.get("total", 0)
                }
                for student_id, data in stats.get("student_stats", {}).items()
            }
        }
        
        return serializable_stats
    except Exception as e:
        logger.error(f"Error in get_class_attendance_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            logger.warning(f"Failed login attempt for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Remove the department check for now, as it's not part of the OAuth2PasswordRequestForm
        # if user["role"] == "student" and user.get("department") != form_data.department:
        #     logger.warning(f"Access denied for user {form_data.username} - department mismatch")
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="Access denied for this department",
        #     )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"], "role": user["role"], "department": user.get("department", "Unknown")},
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "role": user["role"], "department": user.get("department", "Unknown")}
    except Exception as e:
        logger.error(f"Error in login_for_access_token: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add this new endpoint for student registration with face upload
@app.post("/register/student")
async def register_student(
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    student_id: str = Form(...),
    class_id: str = Form(...),
    department: str = Form(...),
    face_image: UploadFile = File(...)
):
    try:
        # Process the face image and store the encoding
        contents = await face_image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")

        temp_path = f"temp_{uuid.uuid4()}.jpg"
        try:
            with open(temp_path, "wb") as f:
                f.write(contents)

            face_encodings = DeepFace.represent(
                img_path=temp_path,
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend="retinaface"
            )
            
            if not face_encodings:
                raise HTTPException(status_code=400, detail="No faces detected in the image")

            face_encoding = face_encodings[0]['embedding']
            face_encoding_b64 = base64.b64encode(np.array(face_encoding).tobytes()).decode('utf-8')

            password_hash = get_password_hash(password)

            # Determine department if not provided
            if not department:
                department = "Electronics and Communication" if class_id.startswith('E') else "Computer Science" if class_id.startswith('C') else "Unknown"

            # Add user with department information
            user_id = add_user(
                email=email,
                password=password_hash,
                role="student",
                name=name,
                student_id=student_id,
                face_encoding=face_encoding_b64,
                class_id=class_id,
                department=department
            )
            
            logger.info(f"Student registered successfully with department: {department}")
            return {"message": "Student registered successfully"}

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error registering student: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def add_user_to_database(email, name, student_id, face_encoding):
    """Add a new user to the database."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (email, name, student_id, face_encoding)
            VALUES (?, ?, ?, ?)
        """, (email, name, student_id, face_encoding))
        conn.commit()

# Add this new endpoint for admin login
@app.post("/admin/login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "test@gmail.com" and form_data.password == "test@123":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username, "role": "admin"}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "role": "admin"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ... (rest of the existing code)

@app.get("/student/stats")
async def get_student_stats(token: str = Depends(oauth2_scheme)):
    try:
        logger.info(f"Received token in get_student_stats: {token[:10]}...")
        payload = decode_token(token)
        logger.info(f"Decoded payload: {payload}")
        
        if not payload:
            logger.error(f"Invalid token: {token[:10]}...")
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        student_email = payload.get("sub")
        logger.info(f"Fetching user data for email: {student_email}")
        user = get_user_by_email(student_email)
        logger.info(f"User data: {user}")
        
        if not user:
            logger.error(f"Student not found for email: {student_email}")
            raise HTTPException(status_code=404, detail="Student not found")

        # Get face image from face_encodings.json
        try:
            with open('face_encodings.json', 'r') as file:
                face_data = json.load(file)
                student_id = user.get('student_id')
                if student_id in face_data and isinstance(face_data[student_id], dict):
                    face_image = face_data[student_id].get('image')
                else:
                    face_image = None
        except Exception as e:
            logger.error(f"Error loading face image: {str(e)}")
            face_image = None

        # Get attendance statistics
        attendance_records = await get_student_attendance(user["student_id"])

        if not attendance_records:
            logger.info(f"No attendance records found for student: {user['student_id']}")
            return {
                "attendance_rate": 0,
                "rate_change": 0,
                "present_classes": 0,
                "total_classes": 0,
                "classes_change": 0,
                "records": [],
                "notifications": [],
                "student_id": user["student_id"],
                "face_image": face_image
            }

        # Calculate statistics
        total_classes = len(attendance_records)
        present_classes = sum(1 for record in attendance_records if record['status'] == 'present')
        attendance_rate = (present_classes / total_classes * 100) if total_classes > 0 else 0

        # Get notifications
        notifications = await get_user_notifications(user["id"])

        result = {
            "attendance_rate": round(attendance_rate, 2),
            "rate_change": 0,
            "present_classes": present_classes,
            "total_classes": total_classes,
            "classes_change": 0,
            "records": attendance_records,
            "notifications": notifications,
            "student_id": user["student_id"],
            "face_image": face_image
        }
        logger.info(f"Returning result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in get_student_stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/attendance/mark/class")
async def mark_class_attendance(
    class_id: str = Form(...),
    date: Optional[str] = Form(None),
    class_photo: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="Not authorized")

        contents = await class_photo.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")

        temp_path = f"temp_{uuid.uuid4()}.jpg"
        try:
            with open(temp_path, "wb") as f:
                f.write(contents)

            # Detect all faces in the class photo
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )

            if not faces:
                raise HTTPException(status_code=400, detail="No faces detected in the image")

            # Get all students in the class
            with get_db() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT student_id, face_encoding 
                    FROM users 
                    WHERE class_id = ? AND face_encoding IS NOT NULL
                """, (class_id,))
                students = c.fetchall()

            # Process each detected face
            marked_students = []
            for face in faces:
                face_img = face['face']
                face_encoding = DeepFace.represent(
                    img_path=face_img,
                    model_name="VGG-Face",
                    enforce_detection=False
                )

                if not face_encoding:
                    continue

                # Compare with all students in the class
                best_match = None
                best_similarity = float('inf')
                for student_id, stored_encoding in students:
                    if stored_encoding:
                        stored_encoding = np.frombuffer(base64.b64decode(stored_encoding), dtype=np.float64)
                        similarity = cosine(face_encoding[0]['embedding'], stored_encoding)
                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_match = student_id

                # Mark attendance if match found
                if best_match and best_similarity < 0.4:  # Adjust threshold as needed
                    attendance_date = date if date else datetime.now().date().isoformat()
                    success, _ = record_attendance(best_match, class_id, attendance_date, "present")
                    if success and best_match not in marked_students:
                        marked_students.append(best_match)

            return {
                "message": "Attendance marked successfully",
                "marked_students": marked_students
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error in mark_class_attendance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def record_attendance(student_id: str, class_id: str, date: date, status: str = "present"):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO attendance (student_id, class_id, date, status)
                VALUES (?, ?, ?, ?)
            """, (student_id, class_id, date, status))
            conn.commit()
            logger.info(f"Attendance recorded for student {student_id} in class {class_id} on {date}")
    except Exception as e:
        logger.error(f"Error recording attendance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record attendance")

    # Send notification
    add_notification(student_id, f"Attendance marked as {status} for class {class_id} on {date}")

@app.get("/attendance/review")
async def review_attendance(class_id: str, date: Optional[str] = None, token: str = Depends(oauth2_scheme)):
    # Validate token and get payload
    payload = decode_token(token)
    if not payload or payload.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Not authorized")

    # Fetch attendance records
    with get_db() as conn:
        c = conn.cursor()
        query = "SELECT * FROM attendance WHERE class_id = ?"
        params = [class_id]
        if date:
            query += " AND date = ?"
            params.append(date)
        c.execute(query, params)
        records = c.fetchall()

    return {"records": records}

@app.post("/attendance/edit")
async def edit_attendance(attendance_id: int, status: str, token: str = Depends(oauth2_scheme)):
    # Validate token and get payload
    payload = decode_token(token)
    if not payload or payload.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Not authorized")

    # Update attendance record
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE attendance SET status = ? WHERE id = ?", (status, attendance_id))
        conn.commit()

    return {"message": "Attendance updated successfully"}

@app.get("/reports/attendance")
async def generate_attendance_report(class_id: str, period: str, token: str = Depends(oauth2_scheme)):
    # Validate token and get payload
    payload = decode_token(token)
    if not payload or payload.get("role") not in ["teacher", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Generate report based on the period
    with get_db() as conn:
        c = conn.cursor()
        if period == "daily":
            c.execute("SELECT date, COUNT(*) FROM attendance WHERE class_id = ? GROUP BY date", (class_id,))
        elif period == "weekly":
            c.execute("SELECT strftime('%W', date) as week, COUNT(*) FROM attendance WHERE class_id = ? GROUP BY week", (class_id,))
        elif period == "monthly":
            c.execute("SELECT strftime('%m', date) as month, COUNT(*) FROM attendance WHERE class_id = ? GROUP BY month", (class_id,))
        else:
            raise HTTPException(status_code=400, detail="Invalid period specified")

        report = c.fetchall()

    return {"report": report}

def add_notification(student_id, message, date=None):
    try:
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        c = get_db().cursor()
        try:
            # Try inserting with date
            c.execute(
                "INSERT INTO notifications (student_id, message, date) VALUES (?, ?, ?)",
                (student_id, message, date)
            )
        except sqlite3.OperationalError as e:
            if "no column named date" in str(e):
                # If date column doesn't exist, insert without it
                c.execute(
                    "INSERT INTO notifications (student_id, message) VALUES (?, ?)",
                    (student_id, message)
                )
        get_db().commit()
        logger.info(f"Added notification for student {student_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding notification: {str(e)}")
        return False

@app.get("/student/attendance")
async def get_student_attendance(token: str = Depends(oauth2_scheme)):
    # Decode the token to get the student ID
    payload = decode_token(token)
    if not payload or payload.get("role") != "student":
        raise HTTPException(status_code=403, detail="Not authorized")

    student_id = payload.get("id")

    # Fetch attendance records for the student
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT class_id, date, status FROM attendance
            WHERE student_id = ?
            ORDER BY date DESC
        """, (student_id,))
        records = c.fetchall()

    return {"attendance_records": records}

@app.get("/debug/token")
async def debug_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        return {"payload": payload}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def get_student_attendance(student_id: str):
    try:
        def fetch_attendance():
            with get_db() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT a.date, a.class_id, a.status
                    FROM attendance a
                    WHERE a.student_id = ?
                    ORDER BY a.date DESC
                """, (student_id,))
                return c.fetchall()

        # Use the connection pool to execute the database query
        rows = await asyncio.get_event_loop().run_in_executor(None, fetch_attendance)
        
        logger.info(f"Fetched {len(rows)} attendance records for student {student_id}")
        
        return [
            {
                "date": str(row[0]),
                "class_id": row[1],
                "status": row[2]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Database error in get_student_attendance: {str(e)}", exc_info=True)
        return []

@app.post("/register_student_for_class")
async def register_student_for_class(data: dict, token: str = Depends(oauth2_scheme)):
    student_id = data.get("student_id")
    class_id = data.get("class_id")
    if not student_id or not class_id:
        raise HTTPException(status_code=400, detail="Missing student_id or class_id")
    
    try:
        # Update the student's class_id in the database
        with get_db() as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET class_id = ? WHERE student_id = ?", (class_id, student_id))
            conn.commit()
        return {"message": f"Student {student_id} registered for class {class_id}"}
    except Exception as e:
        logger.error(f"Error registering student for class: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/attendance/update")
async def update_attendance(
    data: dict = Body(...),  # Use Body to get JSON data
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="Not authorized")

        student_id = data.get("student_id")
        class_id = data.get("class_id")
        date = data.get("date")
        status = data.get("status")

        if not all([student_id, class_id, date, status]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        with get_db() as conn:
            c = conn.cursor()
            
            # Update the attendance record
            c.execute("""
                UPDATE attendance 
                SET status = ? 
                WHERE student_id = ? AND class_id = ? AND date = ?
            """, (status, student_id, class_id, date))
            
            if c.rowcount == 0:
                # If no existing record, insert a new one
                c.execute("""
                    INSERT INTO attendance (student_id, class_id, date, status)
                    VALUES (?, ?, ?, ?)
                """, (student_id, class_id, date, status))
            
            conn.commit()
            
            # Add notification
            try:
                add_notification(student_id, f"Attendance updated to {status} for class {class_id} on {date}")
            except Exception as e:
                logger.error(f"Failed to add notification: {str(e)}")
            
            return {"message": "Attendance updated successfully"}
            
    except Exception as e:
        logger.error(f"Error updating attendance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance/overall_stats")
async def get_overall_attendance_stats(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") not in ["admin", "teacher"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Example logic to aggregate overall statistics
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT status, COUNT(*) as count
                FROM attendance
                GROUP BY status
            """)
            overall_stats = c.fetchall()

        stats_dict = {status: count for status, count in overall_stats}
        return {"overall_stats": stats_dict}
    except Exception as e:
        logger.error(f"Error fetching overall attendance stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Schedule the task to run at 11:59 PM every day
scheduler.add_job(
    mark_absent_students,
    trigger=CronTrigger(hour=23, minute=59),
    id='mark_absent_students'
)

# Start the scheduler when the application starts
@app.on_event("startup")
async def startup_event():
    init_db()
    migrate_db()  # Add this line

# Shutdown the scheduler when the application stops
@app.on_event("shutdown")
async def shutdown_scheduler():
    scheduler.shutdown()

@app.post("/attendance/update_status")  # New endpoint
async def update_attendance_status(
    student_id: str = Form(...),
    class_id: str = Form(...),
    date: str = Form(...),
    status: str = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = decode_token(token)
        if not payload or payload.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="Not authorized")

        with get_db() as conn:
            c = conn.cursor()
            
            # Update the attendance record
            c.execute("""
                UPDATE attendance 
                SET status = ? 
                WHERE student_id = ? AND class_id = ? AND date = ?
            """, (status, student_id, class_id, date))
            
            if c.rowcount == 0:
                # If no existing record, insert a new one
                c.execute("""
                    INSERT INTO attendance (student_id, class_id, date, status)
                    VALUES (?, ?, ?, ?)
                """, (student_id, class_id, date, status))
            
            conn.commit()
            
            return {"message": "Attendance updated successfully"}
            
    except Exception as e:
        logger.error(f"Error updating attendance status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    init_db()  # Initialize the database
    migrate_notifications_table()  # Migrate the notifications table
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
