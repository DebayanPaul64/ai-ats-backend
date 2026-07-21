from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import get_db_connection
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from typing import Literal
import os
import shutil
import jwt
from datetime import datetime, timedelta, timezone
from ai_engine import extract_text_from_pdf, clean_text, generate_tfidf_vectors, calculate_similarity, get_match_percentage

app = FastAPI(title="AI-Powered ATS API")

# Configure CORS to allow frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://debayanpaul64.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow the frontend to access uploaded resumes
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up bcrypt password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

# Verify a plaintext password against the hashed one
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Generate a JWT token valid for 24 hours
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    to_encode.update({"exp": expire})

    # Fetch the secret key from your .env file
    secret_key = os.getenv("JWT_SECRET_KEY", "fallback_secret")

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt

# Tell FastAPI where to find the token (in the Authorization header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get the current logged-in user from the token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        secret_key = os.getenv("JWT_SECRET_KEY", "fallback_secret")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])

        user_id: str = payload.get("sub")
        role: str = payload.get("role")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"user_id": int(user_id), "role": role}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Define the expected data structure for a new user
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Literal['recruiter', 'candidate'] # Restricts input to these two roles

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Define the expected data structure for a new job posting
class JobCreate(BaseModel):
    title: str
    description: str

# Define the expected data structure for updating a job
class JobUpdate(BaseModel):
    title: str
    description: str

@app.get("/")
def read_root():
    return {"message": "CI/CD Test: The magic bridge is officially working!"}
    
# Endpoint to test the database connection
@app.get("/db-check")
def check_db_connection():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        # Create a cursor to execute a simple test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        return {
            "status": "success", 
            "message": "Successfully connected to PostgreSQL!", 
            "postgresql_version": db_version[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/register", status_code=201)
def register_user(user: UserCreate):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        cursor = conn.cursor()
        
        # 1. Check if the user already exists
        cursor.execute("SELECT user_id FROM Users WHERE email = %s;", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered.")
        
        # 2. Hash the plaintext password securely
        hashed_password = get_password_hash(user.password)
        
        # 3. Insert the new user into the database
        insert_query = """
            INSERT INTO Users (name, email, password_hash, role)
            VALUES (%s, %s, %s, %s) RETURNING user_id, name, email, role;
        """
        cursor.execute(insert_query, (user.name, user.email, hashed_password, user.role))
        
        # 4. Fetch the newly created user data and commit the transaction
        new_user = cursor.fetchone()
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            "message": "User registered successfully",
            "user": {
                "user_id": new_user[0],
                "name": new_user[1],
                "email": new_user[2],
                "role": new_user[3]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback() # Undo the insertion if an error occurs
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        cursor = conn.cursor()
        
        # 1. Look up the user by email (Note: the form passes the email inside the 'username' field)
        cursor.execute("SELECT user_id, email, password_hash, role FROM Users WHERE email = %s;", (form_data.username,))
        db_user = cursor.fetchone()
        
        # 2. If user doesn't exist, throw an error
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
            
        # Extract the fields from the database result
        user_id, user_email, hashed_password, user_role = db_user
        
        # 3. Verify the password
        if not verify_password(form_data.password, hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
            
        # 4. Generate the JWT token containing the user's ID and role
        token_data = {"sub": str(user_id), "role": user_role}
        access_token = create_access_token(token_data)
        
        cursor.close()
        conn.close()
        
        # 5. Return the token to the user
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "message": "Login successful!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/jobs", status_code=201)
def create_job(job: JobCreate, current_user: dict = Depends(get_current_user)):
    # 1. Authorization Check: Only recruiters can create jobs
    if current_user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail="Not authorized. Only recruiters can post jobs.")
        
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        cursor = conn.cursor()
        
        # 2. Insert the job into the database, linking it to the recruiter's ID
        insert_query = """
            INSERT INTO Jobs (recruiter_id, title, description)
            VALUES (%s, %s, %s) RETURNING job_id, title, description, created_at;
        """
        cursor.execute(insert_query, (current_user["user_id"], job.title, job.description))
        
        new_job = cursor.fetchone()
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            "message": "Job posted successfully",
            "job": {
                "job_id": new_job[0],
                "title": new_job[1],
                "description": new_job[2],
                "created_at": new_job[3]
            }
        }
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
def get_all_jobs():
    conn = None
    cursor = None
    try:
        # 1. Connection Logic
        conn = get_db_connection() # Update this if your connection function is named differently
        cursor = conn.cursor()
        
        # 2. The Updated Query
        query = """
            SELECT j.job_id, j.title, j.description, j.created_at, u.name AS recruiter_name, j.recruiter_id
            FROM Jobs j
            JOIN Users u ON j.recruiter_id = u.user_id
            ORDER BY j.created_at DESC;
        """
        cursor.execute(query)
        jobs = cursor.fetchall()
        
    except Exception as e:
        # If something goes wrong, tell the frontend exactly what happened
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    finally:
        # 3. Close Logic (Runs no matter what to protect the database)
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
    # 4. Format the Data
    job_list = []
    for job in jobs:
        job_list.append({
            "job_id": job[0],
            "title": job[1],
            "description": job[2],
            "created_at": job[3],
            "recruiter_name": job[4],
            "recruiter_id": job[5]  # The new line that secures your frontend UI
        })
        
    return {"jobs": job_list}
    
@app.put("/jobs/{job_id}")
def update_job(job_id: int, job: JobUpdate, current_user: dict = Depends(get_current_user)):
    # 1. Base Authorization: Must be a recruiter
    if current_user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail="Not authorized. Only recruiters can update jobs.")
        
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        cursor = conn.cursor()
        
        # 2. Ownership Check: Verify the job exists AND belongs to this recruiter
        cursor.execute("SELECT recruiter_id FROM Jobs WHERE job_id = %s;", (job_id,))
        job_record = cursor.fetchone()
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        if job_record[0] != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized. You can only update jobs that you created.")
            
        # 3. Perform the update
        update_query = """
            UPDATE Jobs 
            SET title = %s, description = %s 
            WHERE job_id = %s RETURNING job_id, title, description;
        """
        cursor.execute(update_query, (job.title, job.description, job_id))
        updated_job = cursor.fetchone()
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            "message": "Job updated successfully",
            "job": {
                "job_id": updated_job[0],
                "title": updated_job[1],
                "description": updated_job[2]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/jobs/{job_id}")
def delete_job(job_id: int, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail="Not authorized. Only recruiters can delete jobs.")
        
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        cursor = conn.cursor()
        
        # Ownership Check
        cursor.execute("SELECT recruiter_id FROM Jobs WHERE job_id = %s;", (job_id,))
        job_record = cursor.fetchone()
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        if job_record[0] != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized. You can only delete jobs that you created.")
            
        # Perform the deletion
        cursor.execute("DELETE FROM Jobs WHERE job_id = %s;", (job_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"message": f"Job {job_id} successfully deleted."}
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/applications/{job_id}/upload")
async def upload_resume(job_id: int, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "candidate":
        raise HTTPException(status_code=403, detail="Not authorized. Only candidates can submit applications.")
        
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        cursor = conn.cursor()
        
        # 1. Fetch the job description to compare against the resume
        cursor.execute("SELECT description FROM Jobs WHERE job_id = %s;", (job_id,))
        job_record = cursor.fetchone()
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        job_description = job_record[0]
            
        # 2. Check if the candidate has already applied
        cursor.execute("SELECT application_id FROM Applications WHERE job_id = %s AND candidate_id = %s;", (job_id, current_user["user_id"]))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="You have already applied for this job.")

        # 3. Save the file securely to the server
        os.makedirs("uploads", exist_ok=True)
        safe_filename = f"candidate_{current_user['user_id']}_job_{job_id}.pdf"
        file_location = f"uploads/{safe_filename}"
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # --- NEW AI INTEGRATION WITH BUG FIX ---
        # 4. Run the AI pipeline to calculate the Match Score
        raw_resume = extract_text_from_pdf(file_location)
        clean_resume = clean_text(raw_resume)
        clean_job = clean_text(job_description)
        
        # FIX: Check if the resume had any readable text before doing math
        if not clean_resume.strip():
            final_match_score = 0.0  # Assign 0% if it's an image-only or unreadable PDF
        else:
            matrix = generate_tfidf_vectors(clean_resume, clean_job)
            raw_score = calculate_similarity(matrix)
            final_match_score = get_match_percentage(raw_score)
        # ---------------------------------------
            
        # 5. Record the application AND the ai_match_score in the database
        insert_query = """
            INSERT INTO Applications (resume_file_path, job_id, candidate_id, ai_match_score)
            VALUES (%s, %s, %s, %s) RETURNING application_id, applied_at;
        """
        cursor.execute(insert_query, (file_location, job_id, current_user["user_id"], final_match_score))
        new_app = cursor.fetchone()
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            "message": "Application submitted successfully!",
            "application_id": new_app[0],
            "ai_match_score": final_match_score,
            "applied_at": new_app[1]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/jobs/{job_id}/applications")
def get_job_applications(job_id: int, current_user: dict = Depends(get_current_user)):
    # 1. Base Authorization: Must be a recruiter
    if current_user["role"] != "recruiter":
        raise HTTPException(status_code=403, detail="Not authorized. Only recruiters can view applications.")
        
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        cursor = conn.cursor()
        
        # 2. Ownership Check: Verify the job belongs to this exact recruiter
        cursor.execute("SELECT recruiter_id FROM Jobs WHERE job_id = %s;", (job_id,))
        job_record = cursor.fetchone()
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        if job_record[0] != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized. You can only view applicants for your own jobs.")
            
        # 3. Fetch applications, joined with user data, ranked by AI score!
        query = """
            SELECT a.application_id, u.name, u.email, a.resume_file_path, a.ai_match_score, a.applied_at
            FROM Applications a
            JOIN Users u ON a.candidate_id = u.user_id
            WHERE a.job_id = %s
            ORDER BY a.ai_match_score DESC;
        """
        cursor.execute(query, (job_id,))
        apps = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # 4. Format the output
        app_list = []
        for app in apps:
            app_list.append({
                "application_id": app[0],
                "candidate_name": app[1],
                "candidate_email": app[2],
                # Add the leading slash so the frontend knows to look at the root domain
                "resume_url": f"/{app[3]}", 
                "ai_match_score": float(app[4]) if app[4] else 0.0,
                "applied_at": app[5]
            })
            
        return {"applications": app_list}
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
