"""Seed the mocked BPA database with fake animals.

There is no public Bharat Pashudhan API, so we mock it (and say so
openly in the pitch). Calving dates are computed RELATIVE TO TODAY
so the eligible/ineligible mix stays true no matter when you run
this. Re-run any time: it wipes and reloads the animals collection.

Run:  venv\\Scripts\\python seed.py
"""
from datetime import date, timedelta

from pymongo import MongoClient

# (id, species, breed, lactation_no, days_since_calving, owner, village)
# days_since_calving controls eligibility: NDDB window is day 30-90
# of FIRST lactation. The mix below gives the demo animals to refuse.
ANIMALS = [
    ("356279812345", "cattle",  "Gir",         1,  47, "Ramesh Kumar",    "Anand"),
    ("356279812346", "cattle",  "Sahiwal",     1,  62, "Suresh Patel",    "Anand"),
    ("356279812347", "buffalo", "Murrah",      1,  35, "Vijay Singh",     "Karnal"),
    ("356279812348", "cattle",  "Tharparkar",  1,  88, "Prakash Rao",     "Bikaner"),
    ("356279812349", "buffalo", "Jaffarabadi", 1,  55, "Bharat Desai",    "Junagadh"),
    ("356279812350", "cattle",  "Red Sindhi",  1,  71, "Mohan Lal",       "Hisar"),
    ("356279812351", "buffalo", "Mehsana",     1,  40, "Kiran Chaudhary", "Mehsana"),
    ("356279812352", "cattle",  "Kankrej",     1,  83, "Jagdish Thakor",  "Palanpur"),
    ("356279812353", "buffalo", "Surti",       1,  33, "Nilesh Patel",    "Surat"),
    ("356279812354", "cattle",  "Hariana",     1,  66, "Satish Yadav",    "Rohtak"),
    ("356279812355", "cattle",  "Gir",         1,  45, "Dinesh Rabari",   "Rajkot"),
    ("356279812356", "buffalo", "Murrah",      1,  78, "Rajbir Singh",    "Jind"),
    # --- ineligible: too soon after calving (before day 30) ---
    ("356279812357", "cattle",  "Sahiwal",     1,  12, "Om Prakash",      "Ferozepur"),
    ("356279812358", "buffalo", "Nili Ravi",   1,  21, "Gurmeet Singh",   "Amritsar"),
    # --- ineligible: past the day-90 window ---
    ("356279812359", "cattle",  "Gir",         1, 120, "Kanti Bhai",      "Amreli"),
    ("356279812360", "buffalo", "Banni",       1, 150, "Ismail Jat",      "Bhuj"),
    # --- ineligible: not first lactation ---
    ("356279812361", "cattle",  "Sahiwal",     2,  50, "Balwinder Kaur",  "Ludhiana"),
    ("356279812362", "buffalo", "Murrah",      3,  60, "Ram Niwas",       "Bhiwani"),
    ("356279812363", "cattle",  "Kankrej",     2,  44, "Hitesh Chaudhri", "Patan"),
    ("356279812364", "buffalo", "Pandharpuri", 4,  70, "Sanjay Mane",     "Solapur"),
]


def main():
    client = MongoClient("mongodb://127.0.0.1:27017")
    db = client["sih25005"]

    db.animals.delete_many({})

    docs = []
    today = date.today()
    for aid, species, breed, lact, days_ago, owner, village in ANIMALS:
        calving = today - timedelta(days=days_ago)
        # rough dob: first calving around 3 years old, later ones +14 months each
        dob = calving - timedelta(days=365 * 3 + 425 * (lact - 1))
        docs.append({
            "_id": aid,
            "species": species,
            "breed": breed,
            "sex": "female",
            "dob": dob.isoformat(),
            "lactation_no": lact,
            "last_calving_date": calving.isoformat(),
            "owner": owner,
            "village": village,
        })

    db.animals.insert_many(docs)
    print(f"seeded {db.animals.count_documents({})} animals into sih25005.animals")


if __name__ == "__main__":
    main()
