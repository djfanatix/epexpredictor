from datetime import datetime
from app.predict import run_prediction
from pathlib import Path
import json
import os

OUTPUT = Path("/data/epex_prediction.json")

def allowed_to_run():
    now = datetime.now().strftime("%H:%M")
    return now in ("13:45", "14:15")

def main():
    if not allowed_to_run():
        print("Outside price publication window, skipping.")
        return

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSO-E API key missing")

    result = run_prediction(api_key)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("Prediction updated at", datetime.now())

if __name__ == "__main__":
    main()