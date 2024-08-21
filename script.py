pip install PyPDF2
import argparse
import os
import joblib
import PyPDF2
import sys
import pandas as pd
import numpy as np

# Load the model and necessary components
model = joblib.load('/content/resume_categorizer_model.pkl')
label_encoder = joblib.load('/content/label_encoder.pkl')
tfidf_vectorizer = joblib.load('/content/tfidf_vectorizer.pkl')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def categorize_resumes(input_dir):
    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(input_dir, filename)
            resume_text = extract_text_from_pdf(filepath)

            # Preprocess and predict
            resume_tfidf = tfidf_vectorizer.transform([resume_text])
            predicted_category = model.predict(resume_tfidf)[0]
            predicted_category_label = label_encoder.inverse_transform([predicted_category])[0]

            # Move file to category folder
            category_folder = os.path.join(input_dir, predicted_category_label)
            os.makedirs(category_folder, exist_ok=True)
            os.rename(filepath, os.path.join(category_folder, filename))

            # Store the result for CSV
            results.append([filename, predicted_category_label])

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=['filename', 'category'])
    results_df.to_csv('categorized_resumes.csv', index=False)
    print("Resumes categorized and results saved to categorized_resumes.csv.")

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        input_dir = '/content/drive/MyDrive/CV Screening/Dataset'
        categorize_resumes(input_dir)
    else:
        parser = argparse.ArgumentParser(description='Categorize resumes into respective domains.')
        parser.add_argument('input_dir', type=str, help='Directory containing resume files')
        args = parser.parse_args()

        categorize_resumes(args.input_dir)