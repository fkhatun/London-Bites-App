import json
import os
import pickle
import bcrypt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Streamlit App Config
st.set_page_config(page_title="London Bites", page_icon="🍽️")

# File Paths
USER_DATA_FILE = "users.json"
COLLABORATIVE_MODEL_FILE = "collaborative_model.pkl"
DATASET_PATH = "London dataset.csv"


# Load user data
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            users = json.load(file)
    else:
        users = {}

    for user in users:
        users[user].setdefault("favourites", [])
        users[user].setdefault("saved_recommendations", [])
        users[user].setdefault("ratings", {})
        users[user].setdefault("reviews", {})

    return users


# Save user data
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)

    return load_users()


# Load dataset
@st.cache_data
def load_restaurant_data():
    if not os.path.exists(DATASET_PATH):
        st.error("Dataset file not found.")
        return None

    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.lower()
        if "cuisine" not in df.columns:
            st.error("Error: 'cuisine' column not found.")
            return None
        return df.dropna(subset=["cuisine"])
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


df = load_restaurant_data()


# Content-based recommendation
def recommend_restaurants_content(user_cuisine, top_n=100):
    df = load_restaurant_data()
    if df is None:
        return []

    vectorizer = TfidfVectorizer()
    cuisine_matrix = vectorizer.fit_transform(df["cuisine"].astype(str))
    cosine_sim = cosine_similarity(cuisine_matrix)

    indices = df[df["cuisine"].str.contains(user_cuisine, case=False, na=False)].index.tolist()
    if not indices:
        return []

    scores = sorted(indices, key=lambda i: -cosine_sim[i].sum())[:top_n]
    return df.iloc[scores][["borough", "name", "restaurant_id", "cuisine", "halal", "rating", "user_id", "reviews"]]


# Train collaborative model
def train_collaborative_model():
    df = load_restaurant_data()
    if df is None or not {"borough", "user_id", "restaurant_id", "rating"}.issubset(df.columns):
        return None

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["borough", "user_id", "restaurant_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)

    with open(COLLABORATIVE_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# Load collaborative model
if os.path.exists(COLLABORATIVE_MODEL_FILE):
    with open(COLLABORATIVE_MODEL_FILE, "rb") as f:
        collaborative_model = pickle.load(f)
else:
    st.write("Training collaborative filtering model...")
    collaborative_model = train_collaborative_model()


# Hybrid Recommendation Function
def hybrid_recommend(user_id, user_cuisine, top_n=100):
    df = load_restaurant_data()
    if df is None:
        return []

    content_recommendations = recommend_restaurants_content(user_cuisine, top_n)
    if content_recommendations.empty:
        return []

    content_restaurants = content_recommendations[["name", "restaurant_id"]].values.tolist()
    predictions = []

    for restaurant_name, restaurant_id in content_restaurants:
        if pd.isna(restaurant_id):
            continue
        pred = collaborative_model.predict(user_id, int(restaurant_id)).est
        predictions.append((restaurant_name, pred))

    predictions = sorted(predictions, key=lambda x: -x[1])
    return [(restaurant, pred) for restaurant, pred in predictions[:top_n]]


# ---- Session State Initialization ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

users = load_users()

# ---- Sidebar Navigation ----
st.sidebar.title("🍽️ London Bites")
menu = st.sidebar.radio("Select an option:", ["Login", "Register"])

# ---- LOGIN ----
if menu == "Login":
    st.sidebar.subheader("🔑Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users[username]["user_id"]
            st.sidebar.success(f"✅Welcome, {users[username]['name']}!🎉")
        else:
            st.sidebar.error("❌Invalid username or password!")

# ---- REGISTER ----
elif menu == "📝Register":
    st.sidebar.subheader("Create an Account")

    new_username = st.sidebar.text_input("Choose a Username")
    new_name = st.sidebar.text_input("Full Name")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")

    if st.sidebar.button("Register"):
        if new_username in users:
            st.sidebar.error("Username already exists!")
        elif new_password != confirm_password:
            st.sidebar.error("⚠️Passwords do not match!")
        else:
            user_id = len(users) + 1
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

            users[new_username] = {
                "name": new_name,
                "password": hashed_password,
                "user_id": user_id,
                "favourites": [],
                "saved_recommendations": [],
                "ratings": {},
                "reviews": {},
                "following": {}
            }
            save_users(users)
            st.sidebar.success("🎉Account created! Please log in.")

# ---- LOGOUT ----
if st.session_state["authenticated"]:
    if st.sidebar.button("🚪Logout"):
        st.session_state.clear()
        st.sidebar.success("✅Logged out!")

# ---- Sidebar Menu Navigation ----
menu = st.sidebar.radio("Go to", ["Favourites"], key="main_menu")

# ---- Favourites Page ----
if st.session_state.get("authenticated") and menu == "Favourites":
    st.title("❤️ My Favourite Restaurants")  # Title should always appear

    username = st.session_state["username"]
    user_favourites = users[username].get("favourites", [])

    # Ensure the favourites list is updated from `users.json`
    users = load_users()  # ✅ Reload updated users dictionary
    favourites = users.get(username, {}).get("favourites", [])

    if not user_favourites:
        st.write("You have no favourite restaurants yet.")
    else:
        for fav in user_favourites:
            fav_data = df[df["name"] == fav]

            if fav_data.empty:
                continue

            rating = fav_data["rating"].values[0] if "rating" in fav_data.columns else "N/A"
            borough = fav_data["borough"].values[0] if "borough" in fav_data.columns else "N/A"
            halal = fav_data["halal"].values[0] if "halal" in fav_data.columns else None
            halal_status = "Yes" if halal else "No" if halal is not None else "Not specified"

            # Display restaurant details
            st.write(f"### {fav}")
            st.write(f"📍 **Location**: {borough}")
            st.write(f"⭐ **Rating**: {rating}")
            st.write(f"حَلَال **Halal**: {halal_status}")

            # Remove from favourites button
            if st.button(f"Remove {fav} from Favourites", key=f"remove_{fav}"):
                users[username]["favourites"].remove(fav)
                save_users(users)
                st.success(f"{fav} has been removed from your favourites!")
                st.rerun()
