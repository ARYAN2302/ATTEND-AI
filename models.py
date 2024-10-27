from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: str
    role: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    face_encoding: Optional[str] = None

    class Config:
        orm_mode = True

class AttendanceRecord(BaseModel):
    id: int
    student_id: int
    class_id: str
    timestamp: datetime
    status: str

class AttendanceReport(BaseModel):
    student_email: str
    attendance_count: int
    attendance_percentage: float
    dates_present: List[datetime]

class TeacherRegistration(BaseModel):
    name: str
    email: str
    password: str
