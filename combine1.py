import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string

# Load your training data
cv_data = json.load(open('train_data.json', 'r'))

def spacy_doc(file, data):
    nlp = spacy.blank('en')
    db = DocBin()
    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        annot = annot['entities']

        ents = []
        entity_indices = []

        for start, end, label in annot:
            skip_entity = False
            for idx in range(start, end):
                if idx in entity_indices:
                    skip_entity = True
                    break
            if skip_entity:
                continue

            entity_indices += list(range(start, end))

            try:
                span = doc.char_span(start, end, label=label, alignment_mode='strict')
            except:
                continue

            if span is None:
                err_data = str([start, end]) + "    " + str(text) + "\n"
                file.write(err_data)
            else:
                ents.append(span)

        try:
            doc.ents = ents
            db.add(doc)
        except:
            pass
    return db

# Split the data into training and testing sets
train, test = train_test_split(cv_data, test_size=0.3)

# Open the file with UTF-8 encoding
file = open('train_file.txt', 'w', encoding='utf-8')

# Create spacy DocBin for training data
db_train = spacy_doc(file, train)
db_train.to_disk('train_data.spacy')

# Create spacy DocBin for testing data
db_test = spacy_doc(file, test)
db_test.to_disk('test_data.spacy')

file.close()

import spacy_transformers
nlp = spacy.load('model-best')

#my code
#my code
nlp1 = spacy.load("en_core_web_sm")
sentencizer = nlp1.add_pipe('sentencizer')

stop_words = set(stopwords.words("english"))
categories = ['PI', 'Sum', 'Skill', 'Qc', 'Obj', 'Edu', 'Exp']

train1 = pd.read_csv('data_new.csv', header=0)

# Replace NaN values in 'Text' and 'Category' columns with an empty string and 'Unknown' category
train1.Text.fillna('', inplace=True)
train1.Category.fillna('Unknown', inplace=True)

# Split the data into training and testing sets (if needed)
X_train, X_test, y_train, y_test = train_test_split(train1.Text, train1.Category, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)

classifier = MultinomialNB().fit(X_train_tfidf, y_train)

# Transform the test set
X_test_tf = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)

# Predict the categories for the test set
y_pred = classifier.predict(X_test_tfidf)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n\nClassification Accuracy: {accuracy * 100:.2f}% \n")
#p
#p
import sys,fitz
fname='Uploaded_Resumes/full-stack.pdf'
doc=fitz.open(fname)
text=" "
for page in doc:
    text=text+str(page.get_text())
text

from nltk.corpus import stopwords
doc=nlp(text)
for ent in doc.ents:
  print(ent.text," ---->",ent.label_)

text = text.replace("\n"," ")
from spacy.matcher import PhraseMatcher

# Load the spaCy English model
#nlp1 = spacy.load("en_core_web_sm")

#my code
#my code

pdf_file_path = 'Uploaded_Resumes\\full-stack.pdf' 
pdf = PyPDF2.PdfReader(pdf_file_path)

num_pages = len(pdf.pages)
text1 = ""
for page_number in range(len(pdf.pages)):
    page = pdf.pages[page_number]
    text1 += page.extract_text()

text1 = ''.join(char for char in text1 if char.isascii())

# Remove stopwords using NLTK
text_tokens = text1.split()
filtered_text = [word for word in text_tokens if word.lower() not in stop_words]
filtered_text = ' '.join(filtered_text)

# Segment the text using spaCy
doc1 = nlp(filtered_text)
sentencizer(doc1)
sentences = [sent.text for sent in doc1.sents]
#print("\nsentences:\n", sentences)

# Classify each segment
segment_categories = []
for sentence in sentences:
    X_test_tf = vectorizer.transform(sentences)
    X_test_tfidf = tfidf_transformer.transform(X_test_tf)
    predicted = classifier.predict(X_test_tfidf)
    segment_categories.append(predicted)

#print("\n Categories of segments in text extracted  in " ,pdf_file_path,"is: \n " ,segment_categories)
# Assuming segment_categories is a pandas DataFrame column
segment_categories = classifier.predict(X_test_tfidf)

# Extract segments with category "Skills"
pi_segments = [sentence for sentence, category in zip(sentences, segment_categories) if category == "PI"]

print("\nsummary: \n")
for segment in pi_segments:
    print(segment)

pi_segments = [sentence for sentence, category in zip(sentences, segment_categories) if category == "Exp"]

for segment in pi_segments:
    print(segment)



#p
#p

def extract_skills(text, skills):
    """Extract skills from a given text using spaCy's PhraseMatcher."""
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(skill) for skill in skills]
    matcher.add("SKILL", None, *patterns)

    doc = nlp(text)
    matches = matcher(doc)

    # Extract matched skills from the matches
    matched_skills = [doc[start:end].text for _, start, end in matches]

    return matched_skills

def match_skills(candidate_skills, required_skills):
    """Match candidate skills with required skills and calculate a score."""
    matched_skills = set(candidate_skills) & set(required_skills)

    # Assign a score to each matched skill (for simplicity, score = 1)
    skill_scores = {skill: 1 for skill in matched_skills}

    # Calculate the total score for the candidate
    total_score = sum(skill_scores.values())

    return total_score, skill_scores

# Example job description with required skills
job_description = """
We are seeking a talented and motivated Software Developer to join our dynamic IT team.
The ideal candidate will play a key role in designing, developing, testing, and maintaining high-performance software solutions.
If you are passionate about software development, enjoy working in a collaborative environment,
and have a strong foundation in programming, we want to hear from you.

Programming Skills:
- Primary Language: Proficiency in at least one of the following languages is required:
  - Java
  - Python
  - C++
  - Ruby

Web Development: Experience with front-end and back-end web development technologies:
- HTML
- CSS
- JavaScript
- Web frameworks (e.g., Django, Flask, Spring)

Database Management:
- Knowledge of database systems, including relational databases (e.g., MySQL, PostgreSQL) and NoSQL databases (e.g., MongoDB).

Version Control:
- Experience with version control systems, especially Git.

Preferred Skills:
- Experience with cloud platforms (e.g., AWS, Azure, Google Cloud).
- Knowledge of containerization and orchestration tools (e.g., Docker, Kubernetes).
- Familiarity with Agile/Scrum methodologies.
- Understanding of cybersecurity principles.
- Mobile app development experience (iOS, Android).
"""

# Required skills list
required_skills = ["Java", "Python", "C++", "Ruby", "HTML", "CSS", "JavaScript", "Django", "Flask", "Spring",
                   "MySQL", "PostgreSQL", "MongoDB", "Git", "AWS", "Azure", "Google Cloud", "Docker",
                   "Kubernetes", "Agile", "Scrum", "Cybersecurity", "iOS", "Android"]

# Extract skills from the job description
job_skills = extract_skills(job_description, required_skills)

# Example candidate resume
#candidate_resume = "Experienced Python developer with a focus on machine learning. Strong communication and problem-solving skills."
candidate_skills = extract_skills(doc, required_skills)
candidate_skills = list(set(candidate_skills))

# Match candidate skills with required skills and calculate a score
total_score, skill_scores = match_skills(candidate_skills, job_skills)

low_threshold = 4
intermediate_threshold = 7
high_threshold = 15

# Print the results
#print("\nRequired Skills:", required_skills)
print("\nRequired Job Skills:", job_skills)
print("\nCandidate Skills:", candidate_skills)
print("\nSkill Scores:", skill_scores)
print("\nTotal Score:", total_score/4)

# Determine skill level based on the total score
if total_score < low_threshold:
    print("\nSkill Level: Low")
elif total_score < intermediate_threshold:
    print("\nSkill Level: Intermediate")
else:
    print("\nSkill Level: High")

remaining_skills = set(required_skills) - set(candidate_skills)
print("\nRequired Skills for candidate's improvement:")
print(", ".join(remaining_skills))
