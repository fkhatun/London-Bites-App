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
from geopy.distance import geodesic

# Set up the Streamlit app configuration with a title and icon.
st.set_page_config(page_title="London Bites", page_icon="🍽️")

# Define file paths for user data, collaborative model, and dataset.
USER_DATA_FILE = "users.json"
COLLABORATIVE_MODEL_FILE = "collaborative_model.pkl"
DATASET_PATH = "London dataset.csv"


# Function to load user data from the JSON file.
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            users = json.load(file)  # Load the JSON data into a Python dictionary.
    else:
        users = {}

    # Ensure each user dictionary has required keys initialized.
    for user in users:
        users[user].setdefault("favourites", [])
        users[user].setdefault("saved_recommendations", [])
        users[user].setdefault("ratings", {})
        users[user].setdefault("reviews", {})
    return users


# Function to save user data to the JSON file.
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)  # Save the dictionary as a JSON file with indentation.
    return load_users()  # Reload the users after saving.


# Function to load the restaurant dataset.
@st.cache_data
def load_restaurant_data():
    if not os.path.exists(DATASET_PATH):
        st.error("Dataset file not found.")
        return None

    try:
        df = pd.read_csv(DATASET_PATH)  # Read CSV file into a pandas DataFrame.
        df.columns = df.columns.str.lower()  # Convert column names to lowercase.
        if "cuisine" not in df.columns:
            st.error("Error: 'cuisine' column not found.")
            return None
        return df.dropna(subset=["cuisine"])  # Drop rows where the 'cuisine' column has missing values.
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


# Load the dataset into a DataFrame.
df = load_restaurant_data()

# Prepare options for cuisine and borough dropdowns.
cuisine_options = ["All"] + sorted(df["cuisine"].unique()) if df is not None else ["All"]
borough_options = ["All"] + sorted(df["borough"].unique()) if df is not None else ["All"]


# Function to generate content-based restaurant recommendations based on selected cuisine.
def recommend_restaurants_content(selected_cuisine, top_n=100):
    df = load_restaurant_data()  # Reload the restaurant dataset.
    if df is None:
        return []

    # Create a TF-IDF vectorizer and compute the matrix for the 'cuisine' column.
    vectorizer = TfidfVectorizer()
    cuisine_matrix = vectorizer.fit_transform(df["cuisine"].astype(str))
    # Compute cosine similarity between all restaurant cuisines.
    cosine_sim = cosine_similarity(cuisine_matrix)

    # Find indices where the 'cuisine' contains the selected cuisine (case insensitive).
    indices = df[df["cuisine"].str.contains(selected_cuisine, case=False, na=False)].index.tolist()
    if not indices:
        return []

    # Sort indices by the sum of cosine similarity scores and take top_n recommendations.
    scores = sorted(indices, key=lambda i: -cosine_sim[i].sum())[:top_n]
    # Return a subset of DataFrame columns for the recommended restaurants.
    return df.iloc[scores][["borough", "name", "restaurant_id", "cuisine", "halal", "rating", "lat", "lng"]]


# Function to train the collaborative filtering model.
def train_collaborative_model():
    df = load_restaurant_data()  # Load the restaurant dataset.
    # Check if required columns exist for the collaborative model.
    if df is None or not {"borough", "user_id", "restaurant_id", "rating"}.issubset(df.columns):
        return None

    reader = Reader(rating_scale=(1, 5))
    # Create a Surprise dataset from the DataFrame.
    data = Dataset.load_from_df(df[["borough", "user_id", "restaurant_id", "rating"]], reader)
    # Split the dataset into training and testing sets.
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()  # Initialize the SVD model.
    model.fit(trainset)  # Train the model on the training data.

    # Save the trained model to a file.
    with open(COLLABORATIVE_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# Load the collaborative model from file if it exists, otherwise train a new one.
if os.path.exists(COLLABORATIVE_MODEL_FILE):
    with open(COLLABORATIVE_MODEL_FILE, "rb") as f:
        collaborative_model = pickle.load(f)
else:
    st.write("Training collaborative filtering model...")
    collaborative_model = train_collaborative_model()


# Hybrid recommendation function that combines content-based and collaborative filtering.
def hybrid_recommend(user_id, selected_cuisine, top_n=100):
    df = load_restaurant_data()  # Load the dataset.
    if df is None:
        return []

    # Get content-based recommendations for the selected cuisine.
    content_recommendations = recommend_restaurants_content(selected_cuisine, top_n)
    if content_recommendations.empty:
        return []

    predictions = []
    # Iterate over each recommended restaurant.
    for _, row in content_recommendations.iterrows():
        if pd.isna(row["restaurant_id"]):
            continue
        # Use the collaborative model to predict the rating for the restaurant for this user.
        pred = collaborative_model.predict(user_id, int(row["restaurant_id"])).est
        predictions.append((row, pred))

    # Sort the predictions by the predicted rating in descending order.
    predictions = sorted(predictions, key=lambda x: -x[1])
    return predictions[:top_n]


# ---- Session State Initialization ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Load the users data.
users = load_users()

# ---- Sidebar Navigation ----
st.sidebar.title("🍽️ London Bites")
# Sidebar radio buttons for login and registration.
menu = st.sidebar.radio("Select an option:", ["Login", "Register"])

# ---- LOGIN SECTION ----
if menu == "Login":
    st.sidebar.subheader("🔑 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        # Check if username exists and password matches using bcrypt.
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users[username]["user_id"]
            st.sidebar.success(f"✅ Welcome, {users[username]['name']}! 🎉")
        else:
            st.sidebar.error("❌ Invalid username or password!")

# ---- REGISTER SECTION ----
elif menu == "📝Register":
    st.sidebar.subheader("Create an Account")

    new_username = st.sidebar.text_input("Choose a Username")
    new_name = st.sidebar.text_input("Full Name")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")

    if st.sidebar.button("Register"):
        # Check if the username already exists.
        if new_username in users:
            st.sidebar.error("Username already exists!")
        # Check if both password fields match.
        elif new_password != confirm_password:
            st.sidebar.error("⚠️Passwords do not match!")
        else:
            user_id = len(users) + 1  # Generate a new user ID.
            # Hash the password using bcrypt.
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

            # Create a new user dictionary with all required fields.
            users[new_username] = {
                "name": new_name,
                "password": hashed_password,
                "user_id": user_id,
                "favourites": [],
                "saved_recommendations": [],
                "ratings": {},
                "reviews": {},
                "following": []
            }
            save_users(users)
            st.sidebar.success("🎉Account created! Please log in.")

# ---- LOGOUT SECTION ----
if st.session_state["authenticated"]:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()  # Clear session state on logout.
        st.sidebar.success("✅ Logged out!")

# ---- Search Page Section ----
if st.session_state["authenticated"]:
    st.title("🔍 Search Restaurants")

    # Text input for searching by restaurant name or keyword.
    search_query = st.text_input("Search by restaurant name or keyword")
    # Dropdown for selecting a borough.
    selected_borough = st.selectbox("Select a borough:", borough_options)
    # Dropdown for selecting a cuisine.
    selected_cuisine = st.selectbox("Select a cuisine:", cuisine_options)

    # Inputs for user's current latitude and longitude.
    user_lat = st.number_input("Enter your latitude", value=51.5074)
    user_lng = st.number_input("Enter your longitude", value=-0.1278)

    search_button = st.button("Search")

    if search_button:
        filtered_df = df.copy()

        # Filter the DataFrame based on the search query if provided.
        if search_query:
            filtered_df = filtered_df[filtered_df["name"].str.contains(search_query, case=False, na=False)]
        # Filter by borough if "All" is not selected.
        if selected_borough != "All":
            filtered_df = filtered_df[filtered_df["borough"] == selected_borough]
        # Filter by cuisine if "All" is not selected.
        if selected_cuisine != "All":
            filtered_df = filtered_df[filtered_df["cuisine"] == selected_cuisine]

        if filtered_df.empty:
            st.warning("⚠️ No restaurants found with the selected criteria.")
        else:
            # Iterate over filtered restaurants and calculate distance from user.
            for _, row in filtered_df.iterrows():
                rest_location = (row["lat"], row["lng"])
                user_location = (user_lat, user_lng)
                distance = geodesic(user_location, rest_location).km

                st.write(f"### {row['name']} - ⭐ {row['rating']}")
                st.write(f"📍 *Location**: {row['borough']}")
                st.write(f"🍟 **Cuisine**: {row['cuisine']}")
                st.write(f"📍 Distance: {distance:.2f} km away")
                st.write(f" حَلَال **Halal**: {row.get('halal', 'N/A')}")

        # Get hybrid recommendations using collaborative filtering.
        recommendations = hybrid_recommend(st.session_state["user_id"], selected_cuisine)
        if recommendations:
            st.write("### Recommended Restaurants:")
            # Iterate over recommendations and display details.
            for row, rating in recommendations:
                st.write(f"### {row['name']} - ⭐ {row['rating']}")
                st.write(f"📍 *Location**: {row['borough']}")
                st.write(f"🍟 **Cuisine**: {row['cuisine']}")
                st.write(f" حَلَال **Halal**: {row.get('halal', 'N/A')}")

               
