from fastapi import FastAPI

app = FastAPI(title="SIH25005 Backend")


@app.get("/ping")
def ping():
    return {"status": "ok", "service": "sih25005-server"}
