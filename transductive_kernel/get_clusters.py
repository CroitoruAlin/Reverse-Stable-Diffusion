import joblib

kmeans = joblib.load("kmeans.joblib")
joblib.dump(kmeans.center_cluters_,"clusters.joblib")