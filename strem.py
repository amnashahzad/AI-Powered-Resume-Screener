# Step 1: Import Libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Step 2: Preprocess Text
def preprocess(text):
    """
    Preprocess text by tokenizing, lemmatizing, and removing stopwords/punctuation.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Step 3: Feature Extraction and Ranking
def rank_resume(resume, job_description):
    """
    Compute the similarity score between a resume and a job description using TF-IDF and cosine similarity.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# Step 4: Streamlit App
def main():
    # Set up the Streamlit app
    st.set_page_config(page_title="AI-Powered Resume Screener", page_icon="📄", layout="centered")
    
    # Custom title and description
    st.title("📄 AI-Powered Resume Screener")
    st.markdown("""
        Welcome to the **AI-Powered Resume Screener**!  
        Paste or upload a resume and job description to get a similarity score.  
        The score ranges from **0** (no match) to **1** (perfect match).
    """)
    
    # Use columns to organize inputs
    col1, col2 = st.columns(2)
    
    # Input fields for resume and job description
    with col1:
        st.subheader("Resume")
        resume_text = st.text_area("Paste Resume Here...", height=200, key="resume")
        resume_file = st.file_uploader("Or upload a resume (txt file)", type=["txt"])
    
    with col2:
        st.subheader("Job Description")
        job_description_text = st.text_area("Paste Job Description Here...", height=200, key="job_description")
        job_description_file = st.file_uploader("Or upload a job description (txt file)", type=["txt"])
    
    # Process uploaded files
    if resume_file is not None:
        resume_text = resume_file.read().decode("utf-8")
    if job_description_file is not None:
        job_description_text = job_description_file.read().decode("utf-8")
    
    # Button to compute similarity
    if st.button("🚀 Compute Similarity"):
        if resume_text and job_description_text:
            # Show a progress bar
            with st.spinner("Computing similarity..."):
                time.sleep(1)  # Simulate processing time
                
                # Preprocess the inputs
                resume_processed = preprocess(resume_text)
                job_description_processed = preprocess(job_description_text)
                
                # Compute similarity score
                score = rank_resume(resume_processed, job_description_processed)
                
                # Display the score with interpretation
                st.success(f"✅ **Similarity Score:** {score:.2f}")
                if score < 0.3:
                    st.warning("🔍 **Interpretation:** Low match. Consider revising your resume to better align with the job description.")
                elif score < 0.7:
                    st.info("🔍 **Interpretation:** Medium match. Your resume has some alignment but could be improved.")
                else:
                    st.success("🔍 **Interpretation:** High match! Your resume aligns well with the job description.")
                
                # Allow users to download the result
                result = f"Similarity Score: {score:.2f}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description_text}"
                st.download_button("📥 Download Results", result, file_name="resume_screener_result.txt")
        else:
            st.error("❌ Please provide both a resume and a job description.")
    
    # Reset button
    if st.button("🔄 Reset"):
        st.experimental_rerun()

# Run the Streamlit app
if __name__ == "__main__":
    main()
