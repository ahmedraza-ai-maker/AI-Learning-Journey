import streamlit as st
from analyzer import (
    extract_text,
    detect_skills,
    calculate_score,
    suggest_roles,
    job_match_score,
    match_skills,
    generate_job_recommendation,
    generate_resume_recommendation,
)


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer")

st.write("Upload your resume and let AI analyze it.")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:
    text = extract_text(uploaded_file)

    st.success("Resume uploaded successfully!")

    st.subheader("Extracted Resume Text")

    st.text_area("Resume Text", text, height=300)

    skills = detect_skills(text)

    st.subheader("Detected Skills")

    for skill in skills:
        st.success(skill)

    score = calculate_score(text, skills)

    st.subheader("Resume Score")

    st.progress(score / 100)

    st.write(f"Score: {score}/100")

    resume_recommendations = generate_resume_recommendation(
    score,
    skills
    )

    st.subheader("Resume Recommendation")

    for rec in resume_recommendations:
       st.info(rec)

    

    
    # --------------------------
# Phase 6 Starts Here
# --------------------------

    
    st.subheader("Job Description")

job_description = st.text_area(
    "Paste Job Description Here",
    height=200
)

if job_description:

    st.subheader("Job Description Analysis")

    st.write(job_description)

    job_skills = detect_skills(job_description)

    st.subheader("Required Job Skills")

    for skill in job_skills:
        st.success(skill)

    matched = match_skills(skills, job_skills)

    st.subheader("Matched Skills")

    for skill in matched:
        st.success(skill)
    

    missing_job_skills = []

    for skill in job_skills: 
       if skill not in skills:
         missing_job_skills.append(skill)

    st.subheader("Missing Skills for This Job")

    for skill in missing_job_skills:
        st.error(skill)
    

    job_score = job_match_score(matched, job_skills)

    st.subheader("Job Match Score")

    st.progress(job_score / 100)

    st.write(f"Job Match Score: {job_score}%")
    

    job_recommendations = generate_job_recommendation(
      job_score,
      missing_job_skills

    )
    st.subheader("Job Recommendation")

    for rec in job_recommendations:
       st.info(rec)
    
    roles = suggest_roles(skills)

    st.subheader("Suitable Roles")

    for role in roles:
        st.success(role)