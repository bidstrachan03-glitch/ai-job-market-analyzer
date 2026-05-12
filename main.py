import pandas as pd
import spacy
import matplotlib.pyplot as plt

# Load NLP model
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("data/jobs.csv")

# Skill keywords list (you can expand later)
known_skills = {
    "python", "sql", "aws", "excel", "power bi",
    "java", "c++", "docker", "linux", "react",
    "machine learning", "data analysis"
}

skill_counts = {skill: 0 for skill in known_skills}

# Combine all job descriptions into one loop
for text in df["skills"].dropna():
    doc = nlp(str(text).lower())

    text_content = doc.text

    for skill in known_skills:
        if skill in text_content:
            skill_counts[skill] += 1

# Print results
print("\nAI-Detected Skill Trends:")
for k, v in skill_counts.items():
    print(f"{k}: {v}")

# 📊 Visualization
plt.figure(figsize=(10,5))
plt.bar(skill_counts.keys(), skill_counts.values())

plt.title("AI-Detected Skill Trends (NLP)")
plt.xlabel("Skills")
plt.ylabel("Mentions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()