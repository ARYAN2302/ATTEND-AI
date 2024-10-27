import sqlite3
from datetime import datetime, date, timedelta
from contextlib import contextmanager
from typing import Optional
import logging
from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Create a connection pool
connection_pool = ThreadPoolExecutor(max_workers=5)

@contextmanager
def get_db():
    conn = sqlite3.connect('attendance.db')
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with necessary tables."""
    with sqlite3.connect('attendance.db') as conn:
        c = conn.cursor()
        
        # Create users table with department field
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      email TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT NOT NULL,
                      name TEXT,
                      student_id TEXT UNIQUE,
                      face_encoding TEXT,
                      class_id TEXT,
                      department TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Create attendance table
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      student_id TEXT NOT NULL,
                      class_id TEXT NOT NULL,
                      date DATE NOT NULL,
                      status TEXT NOT NULL,
                      UNIQUE(student_id, class_id, date))''')
        
        # Check if department column exists in users table
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        if 'department' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN department TEXT")
        
        # Check if class_id column exists in users table
        if 'class_id' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN class_id TEXT")
        
        conn.commit()

def migrate_db():
    """Handle any necessary database migrations."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Check if notifications table exists
            c.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='notifications'
            """)
            
            if not c.fetchone():
                # Create notifications table if it doesn't exist
                c.execute('''
                    CREATE TABLE notifications (
                        id INTEGER PRIMARY KEY,
                        student_id TEXT,
                        message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        read INTEGER DEFAULT 0,
                        FOREIGN KEY(student_id) REFERENCES users(student_id)
                    )
                ''')
                conn.commit()
                logger.info("Created notifications table")
            else:
                # Alter table to add student_id if it doesn't exist
                c.execute("PRAGMA table_info(notifications)")
                columns = [column[1] for column in c.fetchall()]
                if 'student_id' not in columns:
                    c.execute("ALTER TABLE notifications ADD COLUMN student_id TEXT")
                    conn.commit()
                    logger.info("Added student_id column to notifications table")
            
    except Exception as e:
        logger.error(f"Error in database migration: {str(e)}")
        raise e

def get_user_by_email(email: str):
    """Get user data by email."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, email, password, role, name, student_id, face_encoding, department 
                FROM users 
                WHERE email = ?
            """, (email,))
            user = c.fetchone()
            
            if user:
                # Create a dictionary with all fields, ensuring department is included
                user_dict = {
                    "id": user[0],
                    "email": user[1],
                    "password": user[2],
                    "role": user[3],
                    "name": user[4],
                    "student_id": user[5] if user[5] is not None else str(user[0]),  # Use user ID as fallback
                    "face_encoding": user[6],
                    "department": user[7] if user[7] is not None else "Unknown"
                }
                logger.info(f"User found: {user_dict}")
                return user_dict
            logger.warning(f"No user found for email: {email}")
            return None
    except Exception as e:
        logger.error(f"Database error in get_user_by_email: {str(e)}", exc_info=True)
        raise

def add_user(email, password, role, name, student_id=None, face_encoding=None, class_id=None, department=None):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (email, password, role, name, student_id, face_encoding, class_id, department)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (email, password, role, name, student_id, face_encoding, class_id, department))
        conn.commit()
        return c.lastrowid

def record_attendance(student_id: str, class_id: str, date: str, status: str = "present"):
    """Record or update attendance."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # First check if record exists
            c.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND class_id = ? AND date = ?
            """, (student_id, class_id, date))
            existing = c.fetchone()
            
            if existing:
                # Update existing record
                c.execute("""
                    UPDATE attendance 
                    SET status = ? 
                    WHERE student_id = ? AND class_id = ? AND date = ?
                """, (status, student_id, class_id, date))
            else:
                # Insert new record
                c.execute("""
                    INSERT INTO attendance (student_id, class_id, date, status)
                    VALUES (?, ?, ?, ?)
                """, (student_id, class_id, date, status))
            
            conn.commit()
            logger.info(f"Successfully {'updated' if existing else 'recorded'} attendance for student {student_id}")
            return True, "updated" if existing else "recorded"
            
    except Exception as e:
        logger.error(f"Error in record_attendance: {str(e)}", exc_info=True)
        return False, str(e)

def get_class_attendance(class_id: str, date: datetime.date):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT a.id, u.student_id, a.status
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE a.class_id = ? AND a.date = ?
        """, (class_id, date))
        return [{"id": row[0], "student_id": row[1], "status": row[2]} for row in c.fetchall()]

def update_attendance_record(record_id: int, status: str):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE attendance SET status = ? WHERE id = ?",
            (status, record_id)
        )
        conn.commit()

async def get_student_attendance(student_id: str):
    """Get attendance records for a specific student."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # First get student details with class_id
            c.execute("""
                SELECT u.name, u.class_id
                FROM users u
                WHERE u.student_id = ?
            """, (student_id,))
            student_info = c.fetchone()
            
            if not student_info:
                logger.warning(f"No student found with ID: {student_id}")
                return None
            
            # Get class_id and determine department
            class_id = student_info[1]
            department = "Electronics and Communication" if class_id.startswith('E') else "Computer Science" if class_id.startswith('C') else "Unknown"
            
            # Get attendance records with proper JOIN
            c.execute("""
                SELECT a.date, a.class_id, a.status
                FROM attendance a
                WHERE a.student_id = ?
                ORDER BY a.date ASC
            """, (student_id,))
            records = c.fetchall()
            
            # Create the response with all required information
            response = {
                "student_info": {
                    "name": student_info[0],
                    "department": department,
                    "class_id": class_id
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
            
            logger.info(f"Returning attendance data for student {student_id}: {response}")
            return response
            
    except Exception as e:
        logger.error(f"Database error in get_student_attendance: {str(e)}", exc_info=True)
        return None

def get_class_stats(class_id: str):
    with get_db() as conn:
        c = conn.cursor()
        
        # Overall stats
        c.execute("""
            SELECT status, COUNT(*) as count
            FROM attendance
            WHERE class_id = ?
            GROUP BY status
        """, (class_id,))
        overall_stats = dict(c.fetchall())
        
        # Student-wise stats
        c.execute("""
            SELECT u.student_id, 
                   SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) as present_count,
                   COUNT(*) as total_count
            FROM attendance a
            JOIN users u ON a.student_id = u.student_id
            WHERE a.class_id = ?
            GROUP BY u.student_id
        """, (class_id,))
        student_stats = {row[0]: {"present": row[1], "total": row[2]} for row in c.fetchall()}
        
        return {
            "overall_stats": overall_stats,
            "student_stats": student_stats
        }

def add_notification(user_id: int, message: str):
    """Add a notification for a user."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO notifications (user_id, message)
                VALUES (?, ?)
            """, (user_id, message))
            conn.commit()
    except Exception as e:
        logger.error(f"Error adding notification: {str(e)}")
        # Don't raise the exception as notifications are not critical
        pass

async def get_user_notifications(user_id: int):
    try:
        def fetch_notifications():
            with get_db() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT id, message, date, 0 as read
                    FROM notifications
                    WHERE student_id = ?
                    ORDER BY date DESC
                """, (user_id,))
                return c.fetchall()

        # Use asyncio to execute the database query
        notifications = await asyncio.get_event_loop().run_in_executor(None, fetch_notifications)
        
        logger.info(f"Fetched {len(notifications)} notifications for user {user_id}")
        
        return [
            {
                "id": n[0],
                "message": n[1],
                "created_at": str(n[2]),
                "read": bool(n[3])
            }
            for n in notifications
        ]
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}", exc_info=True)
        return []

def get_attendance_report(student_id: Optional[int] = None, class_id: Optional[str] = None, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None):
    with get_db() as conn:
        c = conn.cursor()
        query = """
            SELECT u.student_id, u.name, a.class_id, a.date, a.status
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE 1=1
        """
        params = []
        
        if student_id:
            query += " AND u.id = ?"
            params.append(student_id)
        if class_id:
            query += " AND a.class_id = ?"
            params.append(class_id)
        if start_date:
            query += " AND a.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND a.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY a.date DESC, u.name"
        
        c.execute(query, params)
        return [{"student_id": row[0], "name": row[1], "class_id": row[2], "date": row[3], "status": row[4]} for row in c.fetchall()]

def get_class_attendance_range(class_id: str, start_date: datetime.date, end_date: datetime.date):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT a.id, u.student_id, a.date, a.status
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE a.class_id = ? AND a.date BETWEEN ? AND ?
            ORDER BY a.date DESC
        """, (class_id, start_date, end_date))
        
        return [
            {
                "id": row[0],
                "student_id": row[1],
                "date": row[2],
                "status": row[3]
            }
            for row in c.fetchall()
        ]

def check_student_face_encodings():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT student_id, face_encoding FROM users")
        for row in c.fetchall():
            student_id, face_encoding = row
            if not face_encoding:
                print(f"Student {student_id} is missing a face encoding.")

def update_user_face_encoding(user_id: int, face_encoding: str):
    """Update the face encoding for a user."""
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute("""
                UPDATE users 
                SET face_encoding = ?
                WHERE id = ?
            """, (face_encoding, user_id))
            conn.commit()
            logger.info(f"Updated face encoding for user ID: {user_id}")
        except Exception as e:
            logger.error(f"Error updating face encoding: {str(e)}")
            raise Exception(f"Failed to update face encoding: {str(e)}")

def needs_face_encoding_update(user_id: int) -> bool:
    """Check if a user needs their face encoding updated."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT face_encoding 
            FROM users 
            WHERE id = ?
        """, (user_id,))
        result = c.fetchone()
        return not result or not result[0]

def verify_user_table():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='users'
        """)
        table_schema = c.fetchone()[0]
        logger.info(f"Current users table schema: {table_schema}")

def get_user_by_student_id(student_id: str):
    """Retrieve a user from the database by their student ID."""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, email, name, student_id, face_encoding, class_id, department 
                FROM users 
                WHERE student_id = ?
            """, (student_id,))
            row = c.fetchone()
            if row:
                # Determine department based on class_id if not stored
                class_id = row[5]
                stored_department = row[6]
                
                if not stored_department and class_id:
                    if class_id.startswith('E'):
                        department = "Electronics and Communication"
                    elif class_id.startswith('C'):
                        department = "Computer Science"
                    else:
                        department = "Unknown"
                else:
                    department = stored_department or "Unknown"
                
                return {
                    "id": row[0],
                    "email": row[1],
                    "name": row[2],
                    "student_id": row[3],
                    "face_encoding": row[4],
                    "class_id": class_id,
                    "department": department
                }
            return None
    except Exception as e:
        logger.error(f"Error retrieving user by student ID: {str(e)}")
        raise Exception(f"Database error: {str(e)}")

def verify_notifications_table():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='notifications'
        """)
        table_schema = c.fetchone()[0]
        logger.info(f"Current notifications table schema: {table_schema}")

def migrate_notifications_table():
    with sqlite3.connect('attendance.db') as conn:
        c = conn.cursor()
        # Check if the date column exists
        c.execute("PRAGMA table_info(notifications)")
        columns = [column[1] for column in c.fetchall()]
        if 'date' not in columns:
            # Add the date column
            c.execute("ALTER TABLE notifications ADD COLUMN date DATETIME")
        conn.commit()

def fetch_attendance_records(class_id: str, start_date: str, end_date: str):
    logger.info(f"Fetching attendance records for class {class_id} from {start_date} to {end_date}")
    try:
        with get_db() as conn:
            c = conn.cursor()
            # Get attendance records with student details
            c.execute("""
                SELECT 
                    a.student_id, 
                    u.name, 
                    a.date, 
                    a.status,
                    a.class_id
                FROM attendance a
                JOIN users u ON a.student_id = u.student_id
                WHERE a.class_id = ? AND a.date BETWEEN ? AND ?
                ORDER BY u.name, a.date
            """, (class_id, start_date, end_date))
            records = c.fetchall()
            
            logger.info(f"Found {len(records)} attendance records")
            
            department = "Electronics and Communication" if class_id.startswith('E') else "Computer Science" if class_id.startswith('C') else "Unknown"
            
            return [
                {
                    "student_id": record[0],
                    "name": record[1],
                    "department": department,
                    "date": record[2],
                    "status": record[3],
                    "class_id": record[4]
                }
                for record in records
            ]
    except Exception as e:
        logger.error(f"Database error in fetch_attendance_records: {str(e)}", exc_info=True)
        raise

def mark_absent_students():
    """Mark absent for students who weren't marked present today."""
    try:
        today = datetime.now().date()
        with get_db() as conn:
            c = conn.cursor()
            
            # Get all classes and their students
            c.execute("""
                SELECT DISTINCT u.student_id, u.class_id
                FROM users u
                WHERE u.class_id IS NOT NULL
            """)
            class_students = c.fetchall()
            
            # Check each student's attendance
            for student_id, class_id in class_students:
                # Check if attendance was marked today
                c.execute("""
                    SELECT id FROM attendance
                    WHERE student_id = ? AND class_id = ? AND date = ?
                """, (student_id, class_id, today.isoformat()))
                
                if not c.fetchone():
                    # No attendance record found, mark as absent
                    c.execute("""
                        INSERT INTO attendance (student_id, class_id, date, status)
                        VALUES (?, ?, ?, 'absent')
                    """, (student_id, class_id, today.isoformat()))
            
            conn.commit()
            logger.info(f"Marked absent students for {today}")
            
    except Exception as e:
        logger.error(f"Error marking absent students: {str(e)}", exc_info=True)

def migrate_db():
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute('ALTER TABLE users ADD COLUMN department TEXT')
        except sqlite3.OperationalError:
            pass  # Column might already exist
        conn.commit()

# Call this function to ensure the database is initialized with the new schema
init_db()

