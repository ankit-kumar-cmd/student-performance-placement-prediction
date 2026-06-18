import pandas as pd
import numpy as np
import os

def generate_synthetic_data(filename="student_placement.csv", num_students=10000):
    np.random.seed(42)
    
    # 1. Student_ID
    student_ids = [f"STU{i:05d}" for i in range(1, num_students + 1)]
    
    # 2. Attendance (in percentage, between 55% and 98%)
    attendance = np.random.uniform(55, 98, num_students)
    
    # 3. Study_Hours (daily, between 1 and 10 hours)
    study_hours = np.random.uniform(1, 10, num_students)
    
    # 4. CGPA (on a 4.0 scale, correlated with Study_Hours and Attendance)
    # Baseline CGPA of 2.0 + impact of study hours + impact of attendance + noise
    cgpa = 2.0 + (study_hours * 0.15) + ((attendance - 50) * 0.015) + np.random.normal(0, 0.15, num_students)
    cgpa = np.clip(cgpa, 2.0, 4.0) # Cap CGPA between 2.0 and 4.0
    
    # 5. Internal_Marks (out of 100, correlated with CGPA)
    internal_marks = 40 + (cgpa * 12) + np.random.normal(0, 5, num_students)
    internal_marks = np.clip(internal_marks, 50, 100)
    
    # 6. Assignment_Score (out of 100, correlated with Attendance and Study_Hours)
    assignment_score = 35 + (study_hours * 3) + ((attendance - 50) * 0.8) + np.random.normal(0, 6, num_students)
    assignment_score = np.clip(assignment_score, 50, 100)
    
    # 7. Communication_Skills (Categorical: Low, Medium, High)
    # Probability distribution: higher study hours and CGPA correlate with slightly better communication, but mostly random
    comm_choices = ["Low", "Medium", "High"]
    communication_skills = []
    for i in range(num_students):
        # We can add some slight bias based on study hours/cgpa, but keep it mostly categorical
        prob = [0.3, 0.4, 0.3]
        if cgpa[i] > 3.5:
            prob = [0.1, 0.4, 0.5]
        elif cgpa[i] < 2.5:
            prob = [0.5, 0.4, 0.1]
        communication_skills.append(np.random.choice(comm_choices, p=prob))
        
    # 8. Project_Score (out of 100, correlated with CGPA and study hours)
    project_score = 45 + (cgpa * 10) + (study_hours * 2) + np.random.normal(0, 5, num_students)
    project_score = np.clip(project_score, 50, 100)
    
    # 9. Placement_Status (Target Column: Placed or Not Placed)
    # Let's compute a Placement Score to determine Placement Status realistically
    # Placed if: score is high
    comm_numeric = np.array([0 if c == "Low" else 1 if c == "Medium" else 2 for c in communication_skills])
    placement_prob_score = (
        (cgpa - 2.0) * 2.5 + 
        ((attendance - 60) / 10) * 1.2 + 
        (study_hours / 3) * 0.8 + 
        comm_numeric * 0.9 + 
        ((project_score - 50) / 10) * 1.0 +
        np.random.normal(0, 1.0, num_students)
    )
    
    # Threshold for placement
    placement_status = ["Placed" if s >= 4.2 else "Not Placed" for s in placement_prob_score]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance": np.round(attendance, 2),
        "Study_Hours": np.round(study_hours, 2),
        "CGPA": np.round(cgpa, 2),
        "Internal_Marks": np.round(internal_marks, 2),
        "Assignment_Score": np.round(assignment_score, 2),
        "Communication_Skills": communication_skills,
        "Project_Score": np.round(project_score, 2),
        "Placement_Status": placement_status
    })
    
    # Introduce a few random missing values (e.g., 2-3 missing values) to show Phase 2 handling missing values
    # We will introduce them in CGPA and Attendance
    for col in ["CGPA", "Attendance"]:
        missing_indices = np.random.choice(num_students, size=3, replace=False)
        df.loc[missing_indices, col] = np.nan
        
    df.to_csv(filename, index=False)
    print(f"Dataset '{filename}' successfully generated with {num_students} rows.")

if __name__ == "__main__":
    generate_synthetic_data()
