
#  FOOD DELIVERY ANALYSIS PROJECT 
#  Uses: numpy, pandas, matplotlib, seaborn, sklearn
#  Author: Mirin Dixit
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

sns.set(style="whitegrid", palette="Set2")

# ============================================================
# 1) GENERATE SYNTHETIC DATA
# ============================================================

np.random.seed(42)
N = 2000

start = datetime(2024,1,1)
order_times = [start + timedelta(minutes=int(x)) 
               for x in np.random.exponential(scale=60, size=N).cumsum()]

areas = ["Downtown", "Uptown", "Suburb", "TechPark", "University"]
cuisines = ["Indian", "Chinese", "Fast Food", "Italian", "Dessert"]

df = pd.DataFrame({
    "order_id": [f"ORD{i:05d}" for i in range(1, N+1)],
    "customer_id": np.random.randint(1000, 2000, size=N),
    "order_time": order_times,
    "area": np.random.choice(areas, size=N),
    "cuisine": np.random.choice(cuisines, size=N),
    "items": np.random.poisson(2, size=N) + 1,
    "amount": np.round(np.random.normal(300,120,size=N).clip(50,1500),2),
    "delivery_minutes": np.random.randint(15,60,size=N),
    "rating": np.random.choice([3,4,5], size=N, p=[0.2,0.5,0.3])
})

df["weekday"] = df["order_time"].dt.day_name()
df["hour"] = df["order_time"].dt.hour
df["date"] = df["order_time"].dt.date


# ============================================================
# 2) BASIC STATISTICS
# ============================================================

print("\n===== BASIC INFO =====")
print(df.head())
print("\nData shape:", df.shape)
print("\nSummary:")
print(df.describe(include='all'))


# ============================================================
# 3) VISUALIZATIONS
# ============================================================

# ---- Orders by Hour ----
plt.figure(figsize=(10,4))
sns.countplot(x=df["hour"])
plt.title("Orders by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Orders Count")
plt.show()

# ---- Orders by Weekday ----
plt.figure(figsize=(10,4))
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sns.countplot(data=df, x="weekday", order=order)
plt.title("Orders by Weekday")
plt.xticks(rotation=20)
plt.show()

# ---- Cuisine Popularity ----
plt.figure(figsize=(6,6))
df["cuisine"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Cuisine Popularity")
plt.ylabel("")
plt.show()

# ---- Amount vs Area ----
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="area", y="amount")
plt.title("Order Amount by Area")
plt.show()


# ============================================================
# 4) CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(7,4))
sns.heatmap(df[["amount","items","delivery_minutes","rating"]].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 5) SIMPLE FORECASTING (Linear Regression)
# ============================================================

df_daily = df.groupby("date").size().reset_index(name="orders")
df_daily["day_index"] = range(len(df_daily))

X = df_daily[["day_index"]]
y = df_daily["orders"]

model = LinearRegression()
model.fit(X, y)

future_index = np.array([[df_daily["day_index"].max() + i] for i in range(1,8)])
pred = model.predict(future_index).round().astype(int)

print("\n===== NEXT 7 DAYS FORECAST =====")
for i, p in enumerate(pred):
    print(f"Day +{i+1}: {p} orders")


# ============================================================
# 6) CUSTOMER SEGMENTATION (KMeans)
# ============================================================

customer_data = df.groupby("customer_id").agg({
    "order_id":"count",
    "amount":"mean",
    "items":"mean"
}).rename(columns={"order_id":"orders"})

kmeans = KMeans(n_clusters=3, random_state=42)
customer_data["cluster"] = kmeans.fit_predict(customer_data)

print("\n===== CUSTOMER SEGMENTS =====")
print(customer_data.head())

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=customer_data["orders"],
    y=customer_data["amount"],
    hue=customer_data["cluster"],
    palette="Set1"
)
plt.title("Customer Segments")
plt.xlabel("Total Orders")
plt.ylabel("Avg Order Amount")
plt.show()


# ============================================================
# 7) FINAL INSIGHTS
# ============================================================

print("\n===== FINAL INSIGHTS =====")
print("Most popular area:", df["area"].value_counts().idxmax())
print("Most popular cuisine:", df["cuisine"].value_counts().idxmax())
print("Peak hour:", df["hour"].value_counts().idxmax())
print("Peak weekday:", df["weekday"].value_counts().idxmax())
print("Average amount:", df["amount"].mean())
print("============================\n")