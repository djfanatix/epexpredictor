from .elia import load_elia
from .features import build_features
from .entsoe_prices import fetch_day_ahead_prices

def build_training_set(api_key):
    elia = load_elia(history_days=60)
    prices = fetch_day_ahead_prices(api_key)

    df = elia.join(prices, how="inner")
    df = build_features(df)

    X = df.drop(columns=["price_eur_mwh"])
    y = df["price_eur_mwh"]

    return X, y