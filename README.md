# SIH25005 — Image Based Animal Type Classification (Cattle & Buffalo)

Phone app that produces the official NDDB type-classification scorecard
(20 traits, scored 1–9) from photos + one short video. Offline-first,
explainable, linked to the animal's Pashu Aadhaar ear tag.

**Read `SIH25005_Team_Build_Plan.txt` first — it has each person's
features, procedures, and the build order.**

## Folders (commit ONLY inside your own)

| Folder      | Owner    | What lives here                          |
|-------------|----------|------------------------------------------|
| `app/`      | Person 1 | Flutter mobile app                       |
| `ml/`       | Person 2 | Python ML pipeline                       |
| `server/`   | Person 3 | FastAPI backend + MongoDB                |
| `contract/` | frozen   | The JSON contract everyone codes against |

## Daily git habit

- Start of session: `git pull`
- End of session: `git add .` → `git commit -m "what I did"` → `git push`
