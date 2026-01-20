from .training import build_training_set
from .elia import load_elia_forecast
from .model import train_model
from .features import build_features

def run_prediction(api_key):
    X_train, y_train = build_training_set(api_key)
    model = train_model(X_train, y_train)

    elia_forecast = load_elia_forecast()
    X_future = build_features(elia_forecast)

    preds = model.predict(X_future)

    return {
        ts.isoformat(): round(float(p), 2)
        for ts, p in zip(X_future.index, preds)
    }