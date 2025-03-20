# **🚀 AI-Powered Resume Screener – Perfect Your Resume! 💼**
### **📌 Overview**
The AI-Powered Resume Screener helps you match your resume with job descriptions using advanced AI-based analysis. It calculates a Similarity Score and provides detailed feedback to improve your resume, increasing your chances of landing your dream job.

A strong resume isn't just about listing skills – it's about how well your resume matches the job description. This AI tool ensures that your resume highlights the right skills and experiences to make you a perfect fit for the job!

## **🌟 Key Features**
- ✅ AI-based analysis for accurate feedback.
- ✅ Detailed similarity score to evaluate alignment with job requirements.
- ✅ Resume and job description comparison using advanced NLP techniques.
- ✅ User-friendly interface built with Streamlit.
- ✅ Works with all types of resumes and job descriptions.
- ✅ Option to download results for future reference.
- ✅ Actionable suggestions to improve your resume based on the similarity score.
---
## **🔍 How It Works**
- **Upload or Paste Your Resume** – Provide your resume in text format or upload a file.  
- **Upload or Paste Job Description** – Provide the job description in text format or upload a file.  
- **Click "🚀 Compute Similarity"** – AI will analyze your resume and calculate the similarity score.  
- **Get Feedback** – Receive detailed feedback and improvement suggestions.  
- **Download Result** – Save the result for future reference and improvement.  
---
## 📊 Similarity Score Interpretation  

| **Score Range** | **Match Level** | **Feedback** |  
|---------------|----------------|-------------|  
| **0.0 – 0.3** | Low Match       | Your resume needs significant updates to match the job requirements. |  
| **0.3 – 0.7** | Medium Match    | Your resume matches somewhat, but there’s room for improvement. |  
| **0.7 – 1.0** | High Match      | Excellent! Your resume is well aligned with the job requirements. |  
---
## 💻 Installation
Follow these steps to set up the project:

### **1. Clone the Repository**
``` bash
git clone https://github.com/your-username/ai-resume-screener.git
```
### **2. Navigate to the Project Directory**
``` bash
cd ai-resume-screener
```
### **3. Create a Virtual Environment (Optional)**
``` bash
python -m venv venv  
source venv/bin/activate  # Linux/macOS  
.\venv\Scripts\activate   # Windows
```
### **4. Install Dependencies**
``` bash
pip install -r requirements.txt
```
---
## **🚀 Usage**
Run the Application
To launch the Streamlit app, run:

``` bash
streamlit run app.py
```
## **🧠 Example Code for Similarity Calculation**
Here’s a sample code snippet showing how the similarity score is calculated using Cosine Similarity:
``` python
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

def calculate_similarity(resume_text, job_description):  
    vectorizer = TfidfVectorizer()  
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])  
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])  
    return similarity_score[0][0]  

resume = "Experienced software developer with expertise in Python and Machine Learning."  
job_desc = "Looking for a Python developer with a background in Machine Learning and AI."  

score = calculate_similarity(resume, job_desc)  
print(f"Similarity Score: {score:.2f}")
```
---
## **🛠️ Technologies Used**
- **Python** – Backend development
- **Streamlit** – Creating a user-friendly interface
- **Scikit-learn** – Text vectorization and similarity calculation
- **Cosine Similarity** – Comparing resume and job description
- **NLP** – Natural Language Processing for text analysis
---
## **🚩 Future Improvements**
- ✨ Improve NLP model accuracy with fine-tuning.
- ✨ Add support for multiple languages.
- ✨ Introduce AI-generated suggestions for improving resume content.
- ✨ Include keyword-based analysis to highlight missing skills.
- ✨ Allow saving multiple results for better comparison.

## **🤝 Contributing**
Contributions are welcome!

### **Fork the repository**
### **Create a new branch**
``` bash
git checkout -b feature/your-feature
```
### **Commit your changes**
``` bash
git commit -m "Add your feature"
```
### **Push to the branch**
``` bash
git push origin feature/your-feature
```
### **Create a pull request**
---
## **📝 License**
This project is licensed under the MIT License.

## **🌟 Example Output**
Here’s an example of how the similarity score and feedback will appear:
```
Similarity Score: 0.85  
Feedback: Your resume is highly aligned with the job requirements. Great job!
```
## **🌍 Contact**
👩‍💻 Developed by **Amna Shahzad** – Pakistan's Youngest Developer and Data Scientist
<br>
**📧 Email**: coderamna33@gmail.com
<br>
**🔗 GitHub**: https://github.com/amnashahzad
<br>
**🌐 LinkedIn**: https://www.linkedin.com/in/amna-shahzad-data-scientist/
