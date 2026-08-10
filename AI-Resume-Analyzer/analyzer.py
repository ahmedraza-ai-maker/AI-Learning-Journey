from PyPDF2 import PdfReader

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def detect_skills(text):

    skills = [

        # AI / Data Science
        "Python",
        "SQL",
        "NumPy",
        "Pandas",
        "Matplotlib",
        "Scikit-learn",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "NLP",
        "TensorFlow",
        "PyTorch",
        "Data Analysis",
        "Data Visualization",
        "Statistics",
        "Excel",

        # Web Development
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "MongoDB",
        "Bootstrap",
        "Tailwind CSS",
        "Next.js",
        "TypeScript",
        "Express.js",
        "REST API",
        "API",
        "DOM Manipulation",
        "Netlify",
        "Vercel",

        # Mobile Development
        "Flutter",
        "Dart",
        "React Native",
        "Android",
        "iOS",

        # Design
        "Figma",
        "Adobe Photoshop",
        "Illustrator",
        "Canva",
        "UI/UX",

        # Business / Office
        "Microsoft Excel",
        "Power BI",
        "Tableau",
        "Microsoft Word",
        "PowerPoint",
        "Financial Reporting",
        "Bookkeeping",
        "Accounting",
        "QuickBooks",
        "Taxation",

        # HR / Marketing
        "Recruitment",
        "Employee Relations",
        "Human Resources",
        "Communication",
        "Payroll",
        "Social Media Marketing",
        "SEO",
        "Digital Marketing",
        "Content Writing",

        # General Development Tools
        "Git",
        "GitHub",
        "VS Code",
        "Jupyter Notebook",
        "Docker",
        "AWS",
        "Linux"
    ]

    found_skills = []

    text_lower = text.lower()

    for skill in skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    return found_skills

def calculate_score(text, skills):

    score = 0

    # Basic resume sections
    sections = [
        "professional summary",
        "education",
        "experience",
        "projects",
        "skills",
        "certifications"
    ]

    text_lower = text.lower()

    for section in sections:
        if section in text_lower:
            score += 8

    # Skills score
    skill_score = len(skills) * 4

    if skill_score > 40:
        skill_score = 40

    score += skill_score

    # Contact information
    if "@" in text:
        score += 4

    if "linkedin" in text_lower:
        score += 4

    if "github" in text_lower:
        score += 4

    # Limit score
    score = min(score, 100)

    return score

    

def match_skills(resume_skills, job_skills):

    matched = []

    for skill in job_skills:
        if skill in resume_skills:
            matched.append(skill)

    return matched


def job_match_score(matched, job_skills):

    if len(job_skills) == 0:
        return 0

    score = int((len(matched) / len(job_skills)) * 100)

    return score



def generate_recommendation(score, missing):

    recommendations = []

    if score >= 80:
        recommendations.append("Excellent resume. You have a strong AI foundation.")

    elif score >= 60:
        recommendations.append("Good resume, but you can improve it by learning more AI tools.")

    else:
        recommendations.append("Your resume needs more AI-related skills.")

    if "TensorFlow" in missing:
        recommendations.append("Learn TensorFlow for Deep Learning projects.")

    if "Git" in missing:
        recommendations.append("Learn Git and GitHub for version control.")

    if "SQL" in missing:
        recommendations.append("Improve SQL skills for Data Science jobs.")

    if "Machine Learning" in missing:
        recommendations.append("Build Machine Learning projects.")

    return recommendations



def suggest_roles(skills):

    roles = []

    skills_lower = [skill.lower() for skill in skills]

    # AI / Data Science
    if (
        "python" in skills_lower
        or "machine learning" in skills_lower
        or "deep learning" in skills_lower
        or "natural language processing" in skills_lower
        or "nlp" in skills_lower
    ):
        roles.append("AI Intern")
        roles.append("Machine Learning Engineer")
        roles.append("AI Engineer")

    if (
        "data analysis" in skills_lower
        or "pandas" in skills_lower
        or "numpy" in skills_lower
        or "excel" in skills_lower
        or "power bi" in skills_lower
    ):
        roles.append("Data Analyst")

    # Web Development
    if (
        "html" in skills_lower
        or "css" in skills_lower
        or "javascript" in skills_lower
    ):
        roles.append("Frontend Developer")
        roles.append("Web Developer")

    if (
        "react" in skills_lower
        or "node.js" in skills_lower
        or "express.js" in skills_lower
        or "mongodb" in skills_lower
    ):
        roles.append("Full Stack Developer")

    # Mobile Development
    if (
        "flutter" in skills_lower
        or "react native" in skills_lower
        or "android" in skills_lower
    ):
        roles.append("Mobile App Developer")

    # Design
    if (
        "figma" in skills_lower
        or "ui/ux" in skills_lower
    ):
        roles.append("UI/UX Designer")

    if (
        "adobe photoshop" in skills_lower
        or "illustrator" in skills_lower
        or "canva" in skills_lower
    ):
        roles.append("Graphic Designer")

    # HR
    if (
        "recruitment" in skills_lower
        or "human resources" in skills_lower
        or "employee relations" in skills_lower
        or "payroll" in skills_lower
    ):
        roles.append("HR Assistant")
        roles.append("HR Executive")

    # Accounting
    if (
        "accounting" in skills_lower
        or "bookkeeping" in skills_lower
        or "quickbooks" in skills_lower
        or "financial reporting" in skills_lower
    ):
        roles.append("Accountant")
        roles.append("Accounts Assistant")

    # Marketing
    if (
        "digital marketing" in skills_lower
        or "seo" in skills_lower
        or "social media marketing" in skills_lower
        or "content writing" in skills_lower
    ):
        roles.append("Digital Marketing Executive")

    # Default
    if len(roles) == 0:
        roles.append("Entry-Level Professional")
        roles.append("Junior Assistant")
        roles.append("Intern")

    # Remove duplicates
    roles = list(dict.fromkeys(roles))

    return roles


def generate_job_recommendation(match_score, missing_job_skills):

    recommendations = []

    if match_score >= 80:
        recommendations.append(
            "Excellent match for this job. Your resume matches most required skills."
        )

    elif match_score >= 50:
        recommendations.append(
            "Good match, but you should improve some missing skills."
        )

    else:
        recommendations.append(
            "Your resume needs more skills for this specific job."
        )

    for skill in missing_job_skills:
        recommendations.append(f"Consider learning or improving: {skill}")

    return recommendations



def generate_resume_recommendation(score, skills):

    recommendations = []

    if score >= 80:
        recommendations.append(
            "Your resume has a strong structure and good skill coverage."
        )

    elif score >= 60:
        recommendations.append(
            "Your resume is good, but adding more relevant skills, projects, or experience can improve it."
        )

    else:
        recommendations.append(
            "Your resume needs improvement. Add more relevant skills, projects, experience, and professional details."
        )

    if len(skills) < 5:
        recommendations.append(
            "Consider adding more technical or professional skills relevant to your field."
        )

    return recommendations