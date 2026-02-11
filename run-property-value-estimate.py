import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import lightgbm as lgb
import joblib
import os


SAVE_DIR = "./saved_models_strong"
os.makedirs(SAVE_DIR, exist_ok=True)


CATEGORICAL_COLS = [
    "occupancy_status",
    "subject_property",
    "road_class",
    "road_type",
    "area_class"
]

BASE_FEATURES = [
    "lat", "lng", "is_borey",
    "occupancy_status", "subject_property",
    "road_class", "road_type",
    "area_class",
    "branch_sqm_price", "branch_total_price",
    "geo_cluster"   # added later
]

global encoders, models, kmeans_models, small_models, fallback_rows
SKIP_TRAINING = False


def read_excel_with_retry(file_path, categorical_cols, retries=3, delay=2):
    for attempt in range(retries):
        try:
            df = pd.read_excel(file_path)
            df.dropna(subset=['lat','lng','branch_sqm_price','branch_total_price', 'cmu_sqm_price', 'cmu_total_price'])
            df["is_borey"] = df["is_borey"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Failed to read Excel file after {retries} attempts.")


def load_models(file_path):
    global encoders, models, kmeans_models, small_models, fallback_rows
    # Try loading saved models first
    if os.path.exists(f"{file_path}/lgbm_models.pkl"):
        print("Loading saved models...")
        models = joblib.load(f"{file_path}/lgbm_models.pkl")
        kmeans_models = joblib.load(f"{file_path}/kmeans_models.pkl")
        small_models = joblib.load(f"{file_path}/small_models.pkl")
        fallback_rows = joblib.load(f"{file_path}/fallback_rows.pkl")
        encoders = joblib.load(f"{SAVE_DIR}/encoders.pkl")
        SKIP_TRAINING = True
    else:
        print("No saved models found. Training...")
        SKIP_TRAINING = False

    if not SKIP_TRAINING:
        df = read_excel_with_retry(file_path, CATEGORICAL_COLS)

        for poly in df["polygon_name"].unique():
            df_poly = df[df["polygon_name"] == poly].copy()
            df_poly['lat'] = pd.to_numeric(df_poly['lat'], errors='coerce')
            df_poly['lng'] = pd.to_numeric(df_poly['lng'], errors='coerce')
            df_poly['branch_sqm_price'] = pd.to_numeric(df_poly['branch_sqm_price'], errors='coerce')
            df_poly['branch_total_price'] = pd.to_numeric(df_poly['branch_total_price'], errors='coerce')
            df_poly['cmu_sqm_price'] = pd.to_numeric(df_poly['cmu_sqm_price'], errors='coerce')
            df_poly['cmu_total_price'] = pd.to_numeric(df_poly['cmu_total_price'], errors='coerce')

            n = len(df_poly)
            # Not enough data to build a stable model

            if n >= 30:
                # Build KMeans for this polygon only
                n_clusters = max(3, min(15, len(df_poly)//5))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_poly["geo_cluster"] = kmeans.fit_predict(df_poly[["lat", "lng"]])
                kmeans_models[poly] = kmeans

                # Build training matrices
                X = df_poly[BASE_FEATURES]
                y = df_poly["cmu_sqm_price"]

                # Train model
                model = lgb.LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=-1
                )
                model.fit(X, y)

                models[poly] = model
                print(f"Trained model for polygon {poly} with {len(df_poly)} rows")

            elif 10 <= n < 30:
                # Small polygon → reduced LightGBM model (no KMeans)
                X = df_poly[BASE_FEATURES[:-1]]  # remove geo_cluster
                y = df_poly["cmu_sqm_price"]

                small_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.08)
                small_model.fit(X,y)

                small_models[poly] = small_model
                print(f"Trained SMALL model for polygon {poly} ({n} rows)")

            else:
                # Very small → fallback to KNN later
                fallback_rows[poly] = df_poly
                print(f"Polygon {poly} has only {n} rows → using KNN fallback")

        joblib.dump(models, f"{SAVE_DIR}/lgbm_models.pkl")
        joblib.dump(kmeans_models, f"{SAVE_DIR}/kmeans_models.pkl")
        joblib.dump(small_models, f"{SAVE_DIR}/small_models.pkl")
        joblib.dump(fallback_rows, f"{SAVE_DIR}/fallback_rows.pkl")
        joblib.dump(encoders, f"{SAVE_DIR}/encoders.pkl")

        print("All models saved.")


def predict_cmu_price_polygon(lat, lng, polygon_name, **kwargs):
    # 1 — full polygon model
    if polygon_name in models:
        model = models[polygon_name]
        kmeans = kmeans_models[polygon_name]

        sample = pd.DataFrame([{
            "lat": lat,
            "lng": lng,
            **kwargs
        }])

        for col, le in encoders.items():
            if col in sample.columns:
                sample[col] = le.transform([sample[col].iloc[0]])

        sample["geo_cluster"] = kmeans.predict([[lat, lng]])
        return model.predict(sample)[0]

    # 2 — small polygon model
    if polygon_name in small_models:
        sample = pd.DataFrame([{
            "lat": lat,
            "lng": lng,
            **kwargs
        }])

        sample = sample.drop(columns=["geo_cluster"])
        return small_models[polygon_name].predict(sample)[0]

    # 3 — fallback KNN model using neighboring polygons
    if polygon_name in fallback_rows:
        df_small = fallback_rows[polygon_name]

        # compute distance to all points in neighboring polygons
        df_small["dist"] = np.sqrt((df_small["lat"] - lat)**2 +
                                   (df_small["lng"] - lng)**2)

        # use K nearest neighbors
        k = min(3, len(df_small))
        nearest = df_small.nsmallest(k, "dist")

        return nearest["cmu_sqm_price"].mean()

    return "Polygon not found."


if __name__ == "__main__":
    load_models(SAVE_DIR)

    pred = predict_cmu_price_polygon(
        lat=11.49472,
        lng=104.843793,
        polygon_name="12R214",
        is_borey=1,
        occupancy_status="Borrower/Someone Lives in the Property",
        subject_property="Terraced/Linked/Flat house",
        road_class="Sub Road",
        road_type="Concrete Road",
        area_class="Residential Area",
        branch_sqm_price=500,
        branch_total_price=47100
    )

    print(pred)
