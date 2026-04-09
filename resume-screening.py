import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

# ---------- LOAD MODEL ----------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- EXTRACT TEXT FROM PDF ----------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += " " + content
    return text

# ---------- MATCH FUNCTION ----------
def match_resumes(job_desc, resumes):
    scores = []

    # Clean job description
    job_desc_clean = clean_text(job_desc)

    # Extract important keywords
    tfidf = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf.fit([job_desc_clean])
    keywords = tfidf.get_feature_names_out()

    # Encode job description
    job_embedding = model.encode([job_desc_clean])

    for i, resume in enumerate(resumes):
        resume_clean = clean_text(resume)

        # --- Semantic similarity ---
        resume_embedding = model.encode([resume_clean])
        similarity = cosine_similarity(job_embedding, resume_embedding)
        ai_score = similarity[0][0] * 100

        # --- Keyword matching ---
        keyword_matches = [word for word in keywords if word in resume_clean]
        keyword_score = (len(keyword_matches) / len(keywords)) * 30

        # --- Final score ---
        final_score = round(ai_score + keyword_score, 2)

        scores.append((f"Resume {i+1}", final_score, keyword_matches))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Resume Screening", layout="centered")
st.title("📄 AI Resume Screening System")

# Job Description Input
job_desc = st.text_area("Enter Job Description")

# Upload resumes
uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# Button
if st.button("Analyze Candidates"):
    if job_desc and uploaded_files:
        resumes = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)

        results = match_resumes(job_desc, resumes)

        st.subheader("🏆 Candidate Ranking")

        # ✅ Best Candidate
        best = results[0]

        if best[1] >= 75:
            st.success(f"🥇 Best Candidate: {best[0]} → {best[1]}%")
        else:
            st.error("❌ No suitable candidate found (All scores below 75%)")
            st.info("Showing all candidates for reference")

        # ✅ Show all candidates
        for rank, (name, score, keywords) in enumerate(results, 1):
            if score > 85:
                st.success(f"{rank}. {name} → {score}%")
            elif score > 60:
                st.warning(f"{rank}. {name} → {score}%")
            else:
                st.error(f"{rank}. {name} → {score}%")

            st.write(f"Matched Keywords: {keywords}")

    else:
        st.warning("Please enter job description and upload resumes")