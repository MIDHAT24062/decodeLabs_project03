import csv
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "raw_skills.csv"
MIN_SKILLS = 3
TOP_N = 3


def load_roles(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["role"]: row["skills"].split(";") for row in csv.DictReader(f)}


def build_model(roles):
    vectorizer = TfidfVectorizer(analyzer=lambda skills: skills)
    matrix = vectorizer.fit_transform(roles.values())
    return vectorizer, matrix


def match_skills(raw, vocabulary):
    matched, unknown = [], []
    for skill in (s.strip().lower() for s in raw.split(",") if s.strip()):
        if skill in vocabulary:
            found = skill
        else:
            close = difflib.get_close_matches(skill, vocabulary, n=1, cutoff=0.75)
            found = close[0] if close else None
        if found and found not in matched:
            matched.append(found)
        elif not found:
            unknown.append(skill)
    return matched, unknown


def recommend(user_skills, roles, vectorizer, matrix):
    user_vector = vectorizer.transform([user_skills])
    scores = cosine_similarity(user_vector, matrix).ravel()
    ranked = sorted(zip(roles, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:TOP_N]


def main():
    roles = load_roles(DATA_FILE)
    vectorizer, matrix = build_model(roles)
    vocabulary = list(vectorizer.vocabulary_)

    print("Tech Stack Recommender")
    print("Available skills:", ", ".join(sorted(vocabulary)))

    while True:
        raw = input(f"\nEnter at least {MIN_SKILLS} skills, comma separated (or 'q' to quit): ")
        if raw.strip().lower() in ("q", "quit", "exit"):
            break

        skills, unknown = match_skills(raw, vocabulary)
        if unknown:
            print("Not recognised:", ", ".join(unknown))
        if len(skills) < MIN_SKILLS:
            print(f"Need at least {MIN_SKILLS} valid skills, got {len(skills)}.")
            continue

        results = recommend(skills, roles, vectorizer, matrix)

        print("\nYour skills:", ", ".join(skills))
        print(f"Top {TOP_N} matches:")
        for rank, (role, score) in enumerate(results, 1):
            print(f"{rank}. {role} - {score:.0%} match")
            print("   Skills:", ", ".join(roles[role]))


if __name__ == "__main__":
    main()
