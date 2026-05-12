import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Load lightweight English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Load dataset of job postings
df = pd.read_csv("data/jobs.csv")

# Store extracted skill-like phrases
extracted_skills = []

# Loop through all job skill descriptions
for text in df["skills"].dropna():
    # Normalize text for better NLP consistency
    doc = nlp(str(text).lower())

    # Extract meaningful phrases (noun chunks)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()

        # Basic filtering to remove noise
        if 3 <= len(phrase) <= 30:
            extracted_skills.append(phrase)

# Count frequency of extracted phrases
skill_frequency = Counter(extracted_skills)

# Display insights
print("\n🔍 Top Extracted Skill Patterns:\n")

for skill, count in skill_frequency.most_common(15):
    print(f"{skill}: {count}")

# Prepare visualization data
top_skills = skill_frequency.most_common(10)

if top_skills:
    labels, values = zip(*top_skills)

    # Create visualization
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.title("Most Frequent Skill Patterns in Job Postings (NLP Analysis)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.show()
else:
    print("No skills extracted from dataset.")