import json
import os
import pickle
import bcrypt
import pandas as pd
import streamlit as st
import googlemaps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from textblob import TextBlob

# Configure the Streamlit app with a title and an icon.
st.set_page_config(page_title="London Bites", page_icon="🍽️")

# Define file paths for user data, collaborative filtering model, and dataset.
USER_DATA_FILE = "users.json"
COLLABORATIVE_MODEL_FILE = "collaborative_model.pkl"
DATASET_PATH = "London dataset.csv"

# Set your Google API key (make sure to replace with your actual key).
API_KEY = "AIzaSyC7jhTI0x69lbJtp8PaM--nTJ8mg16UgLc"
# Create a Google Maps client using the API key.
gmaps = googlemaps.Client(key=API_KEY)


def get_google_reviews(place_id):
    """
    Fetch restaurant reviews from Google Places API.
    """
    try:
        # Use the Google Maps client to fetch place details, requesting only the 'reviews' field.
        details = gmaps.place(place_id=place_id, fields=["reviews"])
        # Extract reviews from the returned details.
        reviews = details.get("result", {}).get("reviews", [])
        # Debug: Print the entire API response details.
        st.write(f"API response details: {details}")
        # Debug: Print the fetched reviews for the given place_id.
        st.write(f"Reviews for {place_id}: {reviews}")
        # If reviews exist, extract the text of each review and return as a list.
        if reviews:
            return [review["text"] for review in reviews]
        else:
            return []  # Return an empty list if no reviews are found.
    except Exception as e:
        # If an error occurs during API call, display the error in Streamlit.
        st.error(f"Error fetching reviews: {e}")
        return []


# Function to load user data from the JSON file.
def load_users():
    # Check if the user data file exists.
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            users = json.load(file)  # Load the JSON data into a Python dictionary.
    else:
        users = {}  # If the file doesn't exist, start with an empty dictionary.
    # For each user, ensure default keys are present.
    for user in users:
        users[user].setdefault("favourites", [])
        users[user].setdefault("saved_recommendations", [])
        users[user].setdefault("ratings", {})
        users[user].setdefault("reviews", {})
    return users


# Function to save user data to the JSON file.
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)  # Dump the users dictionary into the file with indentation.
    return load_users()  # Reload and return the updated users dictionary.


# Function to load the restaurant dataset from a CSV file.
@st.cache_data
def load_restaurant_data():
    if not os.path.exists(DATASET_PATH):
        st.error("Dataset file not found.")
        return None  # Return None if the dataset file doesn't exist.
    try:
        df = pd.read_csv(DATASET_PATH)  # Read the CSV file into a DataFrame.
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase.
        # Check if the 'cuisine' column is present.
        if "cuisine" not in df.columns:
            st.error("Error: 'cuisine' column not found.")
            return None
        # Return the DataFrame, dropping any rows where the 'cuisine' value is missing.
        return df.dropna(subset=["cuisine"])
    except Exception as e:
        # If an error occurs while loading the dataset, display it in Streamlit.
        st.error(f"Error loading dataset: {e}")
        return None


# Load the restaurant dataset into a DataFrame.
df = load_restaurant_data()


def recommend_restaurants_content(user_cuisine, top_n=100):
    # Reload the restaurant dataset.
    df = load_restaurant_data()
    if df is None:
        return []  # Return an empty list if the dataset couldn't be loaded.

    # Get all unique cuisines available in the dataset (used for debugging/logging here).
    available_cuisines = df["cuisine"].unique()
    st.write(f"Searching for cuisine: {user_cuisine}")

    # Initialize a TfidfVectorizer to transform the 'cuisine' column into TF-IDF features.
    vectorizer = TfidfVectorizer()
    cuisine_matrix = vectorizer.fit_transform(df["cuisine"].astype(str))
    # Calculate cosine similarity between the TF-IDF vectors.
    cosine_sim = cosine_similarity(cuisine_matrix)

    # Get indices of rows where the 'cuisine' contains the user-specified cuisine.
    indices = df[df["cuisine"].str.contains(user_cuisine, case=False, na=False)].index.tolist()
    if not indices:
        st.error(f"No matches found for {user_cuisine}. Try another cuisine.")
        return []  # Return an empty list if no matching cuisine is found.

    # Sort the indices based on the sum of similarity scores and select the top_n indices.
    scores = sorted(indices, key=lambda i: -cosine_sim[i].sum())[:top_n]
    # Return a subset of columns for the recommended restaurants.
    return df.iloc[scores][
        ["borough", "name", "restaurant_id", "cuisine", "halal", "rating", "user_id", "reviews", "place_id"]]


# Function to train a collaborative filtering model using Surprise's SVD.
def train_collaborative_model():
    # Load the restaurant dataset.
    df = load_restaurant_data()
    # Check if the dataset is loaded and contains necessary columns.
    if df is None or not {"borough", "user_id", "restaurant_id", "rating"}.issubset(df.columns):
        return None

    # Create a Reader object with the specified rating scale.
    reader = Reader(rating_scale=(1, 5))
    # Load the data from the DataFrame into a Surprise dataset.
    data = Dataset.load_from_df(df[["borough", "user_id", "restaurant_id", "rating"]], reader)
    # Split the data into training and testing sets.
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()  # Initialize the SVD model.
    model.fit(trainset)  # Train the model on the training set.

    # Save the trained model to a file.
    with open(COLLABORATIVE_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# Load the collaborative filtering model if it exists, otherwise train a new one.
if os.path.exists(COLLABORATIVE_MODEL_FILE):
    with open(COLLABORATIVE_MODEL_FILE, "rb") as f:
        collaborative_model = pickle.load(f)
else:
    st.write("Training collaborative filtering model...")
    collaborative_model = train_collaborative_model()


# Function to perform sentiment analysis on a review using TextBlob.
def analyse_sentiment(review_text):
    """
    Analyzes the sentiment of the review.
    Returns the polarity of the review (positive, neutral, or negative).
    """
    # Create a TextBlob object with the review text.
    blob = TextBlob(review_text)
    # Extract the sentiment polarity from the TextBlob object.
    polarity = blob.sentiment.polarity
    # Return a sentiment label based on the polarity value.
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def get_updated_rating(restaurant):
    """
    Calculate the average rating for a restaurant based on user ratings.
    If no user ratings exist, return None.
    """
    # Create a list of ratings for the restaurant from all users.
    ratings = [users[user]["ratings"].get(restaurant) for user in users if restaurant in users[user]["ratings"]]
    ratings = [r for r in ratings if r is not None]  # Remove any None values from the list.
    # If there are ratings available, compute the average and round it to 2 decimal places.
    if ratings:
        return round(sum(ratings) / len(ratings), 2)
    return None  # Return None if there are no ratings.


def submit_review_and_update_rating(user, restaurant, review_text, rating=None):
    """
    Handles user review submission and updates their rating.
    If a rating is provided, it updates the user's rating for the restaurant.
    """
    # If the user provided a review, save it under the restaurant key.
    if review_text:
        users[user]["reviews"][restaurant] = review_text
    # If the user provided a rating, save it.
    if rating is not None:
        users[user]["ratings"][restaurant] = rating
    # Save the updated user data.
    save_users(users)
    # Return the updated average rating for the restaurant.
    return get_updated_rating(restaurant)


def hybrid_recommend(user_id, user_cuisine, top_n=100):
    # Load the restaurant dataset.
    df = load_restaurant_data()
    if df is None:
        return []

    # Check if the desired cuisine is present in the dataset.
    if user_cuisine.lower() not in df["cuisine"].str.lower().unique():
        st.warning(f"⚠️ No restaurants found for '{user_cuisine}'. Please try a different cuisine.")
        return []

    # Get content-based recommendations for the selected cuisine.
    content_recommendations = recommend_restaurants_content(user_cuisine, top_n)
    if content_recommendations.empty:
        st.warning(f"⚠️ No content-based recommendations found for '{user_cuisine}'.")
        return []

    st.success(f"✅ Found {len(content_recommendations)} content-based recommendations.")

    predictions = []
    # Iterate through each recommended restaurant.
    for _, row in content_recommendations.iterrows():
        restaurant_id = row["restaurant_id"]
        place_id = row["place_id"]
        restaurant_name = row["name"]
        # Skip if restaurant_id or place_id is missing.
        if pd.isna(restaurant_id) or pd.isna(place_id):
            continue
        try:
            # Use the collaborative filtering model to predict a rating for this restaurant.
            pred = collaborative_model.predict(user_id, int(restaurant_id)).est
            # Get Google reviews for the restaurant using its place_id.
            google_reviews = get_google_reviews(place_id)
            # Analyze the sentiment of each Google review.
            sentiments = [analyse_sentiment(review) for review in google_reviews]
            # Append the restaurant name, predicted rating, reviews, and sentiments as a tuple.
            predictions.append((restaurant_name, pred, google_reviews, sentiments))
        except Exception as e:
            st.warning(f"⚠️ Error in collaborative filtering for {restaurant_name}: {e}")
            continue

    # If no predictions were made, warn the user and return content-based results.
    if not predictions:
        st.warning(
            f"⚠️ No collaborative filtering predictions found for '{user_cuisine}', but showing content-based results.")
        return [(row["name"], None, [], []) for _, row in list(content_recommendations.iterrows())[:top_n]]

    # Sort the predictions by predicted rating in descending order.
    predictions = sorted(predictions, key=lambda x: -x[1])
    return predictions[:top_n]


# ---- Session State Initialization ----
# Initialize session state variables if not already set.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Reload users from file.
users = load_users()

# ---- Sidebar Navigation ----
st.sidebar.title("🍽️ London Bites")
# Create a sidebar radio selection for "Login" and "Register".
menu = st.sidebar.radio("Select an option:", ["Login", "Register"])

# ---- LOGIN SECTION ----
# login details
if menu == "Login":
    st.sidebar.subheader("🔑Login")
    # Input field for username.
    username = st.sidebar.text_input("Username")
    # Input field for password (hidden).
    password = st.sidebar.text_input("Password", type="password")
    # Button to trigger login.
    login_button = st.sidebar.button("Login")
    if login_button:
        # Check if username exists and if the password matches using bcrypt.
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            # Update session state variables to mark the user as authenticated.
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users[username]["user_id"]
            st.sidebar.success(f"✅Welcome, {users[username]['name']}!🎉")
        else:
            st.sidebar.error("❌Invalid username or password!")

# ---- REGISTER SECTION ----
elif menu == "Register":
    st.sidebar.subheader("📝 Create an Account")
    # Input fields for new user registration.
    new_username = st.sidebar.text_input("Choose a Username")
    new_name = st.sidebar.text_input("Full Name")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")
    if st.sidebar.button("Register"):
        # Reload users to ensure the latest data.
        users = load_users()
        # Validate that all fields are provided.
        if not new_username or not new_name or not new_password or not confirm_password:
            st.sidebar.error("⚠️ All fields are required!")
        elif new_username in users:
            st.sidebar.error("❌ Username already exists! Choose a different one.")
        elif new_password != confirm_password:
            st.sidebar.error("⚠️ Passwords do not match!")
        else:
            try:
                # Generate a new user_id.
                user_id = len(users) + 1
                # Hash the password.
                hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                # Create a new user entry with all required fields.
                users[new_username] = {
                    "name": new_name,
                    "password": hashed_password,
                    "user_id": user_id,
                    "favourites": [],
                    "saved_recommendations": [],
                    "ratings": {},
                    "reviews": {},
                    "following": []  # Initialize following as an empty list.
                }
                # Save the updated users data.
                save_users(users)
                st.sidebar.success("🎉 Account created! Please log in.")
            except Exception as e:
                st.sidebar.error(f"Error creating account: {e}")

# ---- LOGOUT SECTION ----
if st.session_state["authenticated"]:
    # Provide a logout button if the user is authenticated.
    if st.sidebar.button("🚪Logout"):
        st.session_state.clear()  # Clear session state to log out.
        st.sidebar.success("✅Logged out!")

# ---- Sidebar Menu Navigation for Home Page ----
menu = st.sidebar.radio("Go to", ["Home"], key="main_menu")

# ---- HOME PAGE SECTION ----
if st.session_state["authenticated"]:
    if menu == "Home":
        st.title("Find Your Next Meal 🍽️")
        # If the dataset is loaded and has a 'cuisine' column, get unique cuisines.
        if df is not None and "cuisine" in df.columns:
            cuisines = sorted(df["cuisine"].dropna().unique())
        else:
            cuisines = []
        st.write("### Choose Your Cuisine:")
        # Provide a dropdown for selecting a cuisine.
        selected_cuisine = st.selectbox("Select Cuisine", cuisines, index=None, placeholder="Choose a cuisine")
        if selected_cuisine:
            st.write(f"### 🍽️ Restaurants serving {selected_cuisine}:")
            # Get hybrid recommendations for the selected cuisine.
            recommendations = hybrid_recommend(st.session_state["user_id"], selected_cuisine)
            if not recommendations:
                st.warning("⚠️No restaurants found for this cuisine.")
            else:
                # Loop through each recommended restaurant.
                for restaurant, rating, google_reviews, sentiments in recommendations:
                    # Get additional restaurant details from the dataset.
                    restaurant_data = df[df["name"] == restaurant]
                    borough = restaurant_data["borough"].values[0] if "borough" in restaurant_data.columns else "N/A"
                    halal = restaurant_data["halal"].values[0] if "halal" in restaurant_data.columns else None
                    review_text = restaurant_data["reviews"].values[
                        0] if "reviews" in restaurant_data.columns else "No reviews available."
                    # Display restaurant details.
                    st.write(f"### {restaurant}")
                    st.write(f"📍 **Location**: {borough}")
                    st.write(f"حَلَال **Halal**: {halal}")
                    # Add to favourites button.
                    if st.button(f"Add {restaurant} to Favourites ❤️", key=f"fav_{restaurant}"):
                        if restaurant not in users[st.session_state["username"]]["favourites"]:
                            users[st.session_state["username"]]["favourites"].append(restaurant)
                            save_users(users)
                            st.success(f"{restaurant} has been added to your favourites!")
                        else:
                            st.warning(f"{restaurant} is already in your favourites")
                    # Get the latest rating for the restaurant based on user ratings.
                    updated_rating = get_updated_rating(restaurant)
                    st.write(
                        f"### {restaurant} - Current Rating: {updated_rating if updated_rating else 'No ratings yet'}⭐")
                    # Review submission section.
                    st.write("### Leave a Review")
                    review_input = st.text_area(f"Your review for {restaurant}", key=f"review_{restaurant}")
                    user_rating = st.slider("Rate this restaurant (1-5):", min_value=1, max_value=5, value=3,
                                            key=f"rating_{restaurant}")
                    # Button to submit the review and rating.
                    if st.button(f"Submit review for {restaurant}", key=f"submit_review_{restaurant}"):
                        if st.session_state["username"] in users:
                            user = st.session_state["username"]
                            updated_rating = submit_review_and_update_rating(user, restaurant, review_input,
                                                                             user_rating)
                            st.success(f"Review submitted! Updated Rating: {updated_rating}⭐")
                        else:
                            st.warning("You must be logged in to submit a review.")
                    # Display reviews from other users.
                    st.write("### Reviews from Other Users:")
                    other_reviews = [
                        f"**{users[user]['name']}** ({users[user]['ratings'].get(restaurant, 'No rating')}⭐): {users[user]['reviews'][restaurant]}"
                        for user in users if restaurant in users[user].get("reviews", {})
                    ]
                    if other_reviews:
                        for review in other_reviews:
                            st.write(review)
                    else:
                        st.write("No reviews from other users yet.")
                    # Display Google Reviews with sentiment.
                    if google_reviews:
                        st.write("### Google Reviews with Sentiment:")
                        for review, sentiment in zip(google_reviews, sentiments):
                            st.write(f"**{sentiment.capitalize()} Review**: {review}")
                    else:
                        st.write("No Google reviews available for this restaurant.")
