def build_features(df):
    df = df.copy()

    df["renewable"] = df["pv"] + df["wind"]
    df["renewable_share"] = df["renewable"] / df["load"]
    df["net_import_share"] = df["net_import"] / df["load"]

    for col in ["renewable_share", "wind"]:
        df[f"{col}_lag24"] = df[col].shift(24)

    return df.dropna()