# decodeLabs_project03
# Tech Stack Recommender

DecodeLabs AI Project 3. Recommends job roles based on the skills you enter, using TF-IDF and cosine similarity (content-based filtering).

## Run

```
pip install scikit-learn
python tech_stack_recommender.py
```

Enter at least 3 skills, for example `python, cloud computing, automation`. The script prints the top 3 matching roles with a match percentage.

## How it works

1. Each role in `raw_skills.csv` is an item described by its skills
2. Skills are converted to TF-IDF vectors, so rare skills count more than common ones like `python`
3. Your skills become a vector in the same space
4. Cosine similarity scores every role, then results are sorted and cut to the top 3

## Notes

- Requiring 3 skills up front avoids the cold start problem
- Small typos are corrected (`machine lerning` becomes `machine learning`)
- To add a role, add a row to `raw_skills.csv` with skills separated by `;`
