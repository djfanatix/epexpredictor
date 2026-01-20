import requests
import pandas as pd
from datetime import datetime, timedelta

BASE = "https://web-api.tp.entsoe.eu/api"

def fetch_day_ahead_prices(api_key):
    tomorrow = datetime.utcnow().date() + timedelta(days=1)

    params = {
        "securityToken": api_key,
        "documentType": "A44",          # day-ahead prices
        "in_Domain": "10YBE----------2",
        "out_Domain": "10YBE----------2",
        "periodStart": tomorrow.strftime("%Y%m%d0000"),
        "periodEnd": tomorrow.strftime("%Y%m%d2300"),
    }

    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()

    return parse_entsoe_xml(r.text)

def parse_entsoe_xml(xml):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    ns = {"ns": root.tag.split("}")[0].strip("{")}

    records = []

    for point in root.findall(".//ns:Point", ns):
        pos = int(point.find("ns:position", ns).text)
        price = float(point.find("ns:price.amount", ns).text)

        records.append((pos, price))

    start = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    index = [start + pd.Timedelta(hours=p-1) for p, _ in records]

    return pd.Series(
        [p for _, p in records],
        index=pd.DatetimeIndex(index, tz="UTC").tz_convert("Europe/Brussels"),
        name="price_eur_mwh"
    )