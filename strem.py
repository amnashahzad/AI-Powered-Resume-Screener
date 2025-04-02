import streamlit as st
import time
import spacy
from spacy.cli import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io

# Cache the spaCy model for efficient re-use
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Preprocess text: tokenization, lemmatization, and removal of stopwords/punctuation
def preprocess(text):
    """
    Preprocess text by tokenizing, lemmatizing, and removing stopwords/punctuation.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# File reading functions for TXT, PDF, and DOCX
def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def read_docx(file):
    try:
        document = docx.Document(file)
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def extract_text(file, file_type):
    if file_type == "txt":
        return read_txt(file)
    elif file_type == "pdf":
        return read_pdf(file)
    elif file_type == "docx":
        return read_docx(file)
    else:
        st.error("Unsupported file type.")
        return ""

# Compute similarity score using TF-IDF with custom parameters
def rank_resume(resume, job_description):
    """
    Compute the similarity score between a resume and a job description using TF-IDF and cosine similarity.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0], vectorizer.get_feature_names_out()

# Find top matching words (intersection between important tokens)
def top_matching_words(resume_proc, jd_proc, top_n=5):
    resume_set = set(resume_proc.split())
    jd_set = set(jd_proc.split())
    common = resume_set.intersection(jd_set)
    return sorted(common)[:top_n] if common else []

# Main Streamlit application
def main():
    st.set_page_config(page_title="AI-Powered Resume Screener", page_icon="📄", layout="centered")
    
    st.title("📄 AI-Powered Resume Screener")
    st.markdown("""
        Upload or paste a resume and job description to get a similarity score.  
        The score ranges from **0** (no match) to **1** (perfect match).  
        You can also view the top matching keywords between the two.
    """)

    # Create tabs for Resume and Job Description input
    tabs = st.tabs(["Resume", "Job Description"])
    
    with tabs[0]:
        st.subheader("Resume Input")
        resume_text_input = st.text_area("Paste Resume Here...", height=200, key="resume_text")
        resume_file = st.file_uploader("Or upload a resume file (txt, pdf, docx)", type=["txt", "pdf", "docx"], key="resume_file")
    
    with tabs[1]:
        st.subheader("Job Description Input")
        jd_text_input = st.text_area("Paste Job Description Here...", height=200, key="jd_text")
        jd_file = st.file_uploader("Or upload a job description file (txt, pdf, docx)", type=["txt", "pdf", "docx"], key="jd_file")
    
    # Process uploaded files if available (priority over pasted text)
    if resume_file is not None:
        file_type = resume_file.name.split(".")[-1].lower()
        resume_text = extract_text(resume_file, file_type)
    else:
        resume_text = resume_text_input
    
    if jd_file is not None:
        file_type = jd_file.name.split(".")[-1].lower()
        jd_text = extract_text(jd_file, file_type)
    else:
        jd_text = jd_text_input
    
    # Button to compute similarity
    if st.button("🚀 Compute Similarity"):
        if resume_text.strip() and jd_text.strip():
            with st.spinner("Processing texts and computing similarity..."):
                time.sleep(0.5)  # Minor delay for user feedback
                
                # Preprocess both texts
                resume_processed = preprocess(resume_text)
                jd_processed = preprocess(jd_text)
                
                # Compute similarity score and retrieve feature names (if needed for further analysis)
                score, features = rank_resume(resume_processed, jd_processed)
                
                # Display the similarity score with interpretation
                st.success(f"✅ **Similarity Score:** {score:.2f}")
                if score < 0.3:
                    st.warning("🔍 **Interpretation:** Low match. Consider revising your resume to better align with the job description.")
                elif score < 0.7:
                    st.info("🔍 **Interpretation:** Medium match. Your resume has some alignment but could be improved.")
                else:
                    st.success("🔍 **Interpretation:** High match! Your resume aligns well with the job description.")
                
                # Show top matching keywords from the preprocessed texts
                common_words = top_matching_words(resume_processed, jd_processed, top_n=5)
                if common_words:
                    st.markdown("### Top Matching Keywords")
                    st.write(", ".join(common_words))
                
                # Provide download of the result summary
                result_summary = (
                    f"Similarity Score: {score:.2f}\n\n"
                    f"---\n\n"
                    f"Resume (original):\n{resume_text}\n\n"
                    f"Job Description (original):\n{jd_text}\n\n"
                    f"Top Matching Keywords: {', '.join(common_words) if common_words else 'None'}"
                )
                st.download_button("📥 Download Results", result_summary, file_name="resume_screener_result.txt")
        else:
            st.error("❌ Please provide both a resume and a job description.")
    
    # Reset option to refresh the app
    if st.button("🔄 Reset"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
