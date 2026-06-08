import PyPDF2
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path: str) -> str:
    """
    Opens a PDF file and extracts all readable text.
    Returns the extracted text as a single string.
    """
    # Check if the file actually exists before trying to read it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    try:
        raw_text = ""
        # Open the file in 'rb' (read binary) mode, which is required for PDFs
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Iterate through every page in the document
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # If text was found on the page, add it to our main string
                if text:
                    raw_text += text + "\n"
                    
        return raw_text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def clean_text(raw_text: str) -> str:
    """
    Standardizes text by converting to lowercase, removing punctuation, 
    filtering out stop words, and extracting the root word (lemma).
    """
    if not raw_text:
        return ""
        
    # 1. Convert everything to lowercase
    text_lower = raw_text.lower()
    
    # 2. Process the text through the spaCy English model
    doc = nlp(text_lower)
    
    cleaned_words = []
    
    # 3. Iterate through every word (token) in the document
    for token in doc:
        # Filter out common stop words (like 'the', 'is', 'and'), punctuation, and pure whitespace
        if not token.is_stop and not token.is_punct and token.text.strip():
            # 4. Extract the 'lemma' (the base dictionary form of the word)
            # We also ensure we only keep alphabetical characters (filtering out random bullet points/numbers)
            if token.lemma_.isalpha():
                cleaned_words.append(token.lemma_)
                
    # 5. Join the cleaned words back together into a single string separated by spaces
    return " ".join(cleaned_words)

def generate_tfidf_vectors(resume_text: str, job_description: str):
    """
    Converts the cleaned resume text and job description into TF-IDF vectors.
    Returns the mathematical matrix representing both documents.
    """
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()
    
    # Bundle the two texts together into a list (a "corpus")
    # Index 0 will be the resume, Index 1 will be the job description
    corpus = [resume_text, job_description]
    
    # Fit the vectorizer to the texts and transform them into vectors
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix

def calculate_similarity(tfidf_matrix) -> float:
    """
    Takes a TF-IDF matrix containing two documents and calculates 
    the Cosine Similarity between them.
    Returns a float representing the raw score (0.0 to 1.0).
    """
    # Compare the first document (Index 0) against the second document (Index 1)
    # This returns a 2D array, so we extract the single float value at [0][0]
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return float(similarity_score)

def get_match_percentage(raw_score: float) -> float:
    """
    Converts the raw cosine similarity decimal (0.0 to 1.0) 
    into a clean, user-friendly percentage rounded to two decimal places.
    """
    # Multiply by 100 and round to 2 decimal places (e.g., 0.0795... becomes 7.95)
    percentage = round(raw_score * 100, 2)
    return percentage