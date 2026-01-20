import requests
import pandas as pd

BASE = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"

def fetch(dataset):
    url = f"{BASE}/{dataset}/records"
    r = requests.get(url, params={"limit": 200})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["results"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)\
        .dt.tz_convert("Europe/Brussels")
    return df.set_index("datetime")

def load_elia():
    pv = fetch("ods032")[["forecast"]].rename(columns={"forecast": "pv"})
    wind = fetch("ods086")[["forecast"]].rename(columns={"forecast": "wind"})
    load = fetch("ods003")[["forecast"]].rename(columns={"forecast": "load"})
    flows = fetch("ods021")
    flows["net_import"] = flows["import"] - flows["export"]

    df = pv.join(wind).join(load).join(flows[["net_import"]])
    return df.dropna()