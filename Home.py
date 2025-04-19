import json  # working with user data
import os  # interacting with the file system
import pickle  # saving/ loading serialised python objects
import bcrypt  # securely hashing and verifying user passwords
import pandas as pd  # data manipulation and analysis
import streamlit as st  # building London Bites app
import googlemaps  # accessing GoogleMaps API
from sklearn.feature_extraction.text import TfidfVectorizer  # converting text
from sklearn.metrics.pairwise import cosine_similarity  # calculating similarity between TF-IDF vectors
from surprise import Dataset, Reader, SVD  # building recommendation model based on user ratings using SVD
from surprise.model_selection import train_test_split  # splitting rating data into training and testing sets
from textblob import TextBlob  # sentiment analysis on user reviews

# Configure the Streamlit app with a title and an icon.
st.set_page_config(page_title="London Bites", page_icon="🍽️")

# Defining file paths for user data, collaborative filtering model, and dataset.
USER_DATA_FILE = "users.json"
COLLABORATIVE_MODEL_FILE = "collaborative_model.pkl"
DATASET_PATH = "London dataset.csv"

# Google API key
API_KEY = "AIzaSyC7jhTI0x69lbJtp8PaM--nTJ8mg16UgLc"
gmaps = googlemaps.Client(key=API_KEY)  # Creating a Google Maps client using the API key.


def get_google_reviews(place_id):
    #  Fetch restaurant reviews from Google Places API.
    try:
        details = gmaps.place(place_id=place_id, fields=["reviews"])  # Use the Google Maps client to fetch place details, requesting only the 'reviews' field.
        reviews = details.get("result", {}).get("reviews", [])  # Extract reviews from the returned details.
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
        st.error(f"Error fetching reviews: {e}")   # If an error occurs during API call, display the error in Streamlit.
        return []


# Function to load user data from the JSON file.
def load_users():  # Loads user data from the JSON file and ensures default keys exist.
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
    with open(USER_DATA_FILE, "w") as file:  # Open the user data file (USER_DATA_FILE) in write mode. This will overwrite the existing file with the updated user data
        json.dump(users, file, indent=4)  # Convert the 'users' dictionary into JSON format and write it to the file
        # The 'indent=4' makes the JSON output nicely formatted and readable
    return load_users()  # Reload and return the updated dictionary.


# Function to load the restaurant dataset from a CSV file.
@st.cache_data
def load_restaurant_data():
    if not os.path.exists(DATASET_PATH):
        st.error("Dataset file not found.")
        return None  # Return None if the dataset file doesn't exist.
    try:
        df = pd.read_csv(DATASET_PATH)  # Load the dataset into a pandas DataFrame
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase.
        # checking that the 'cuisine' column exists in the dataset
        if "cuisine" not in df.columns:
            st.error("Error: 'cuisine' column not found.")
            return None
        # Return the DataFrame, dropping any rows where the 'cuisine' value is missing.
        return df.dropna(subset=["cuisine"])
    except Exception as e:
        # Handle any unexpected errors during file reading
        st.error(f"Error loading dataset: {e}")
        return None


# Load the restaurant dataset into a DataFrame.
df = load_restaurant_data()


def recommend_restaurants_content(user_cuisine, top_n=100):
    df = load_restaurant_data()  # Reload the restaurant dataset.
    if df is None:
        return []  # Return an empty list if the dataset couldn't be loaded.

    # Get all unique cuisines available in the dataset.
    available_cuisines = df["cuisine"].unique()
    st.write(f"Searching for cuisine: {user_cuisine}")

    vectorizer = TfidfVectorizer() # Initialise a TfidfVectorizer to transform the 'cuisine' column into TF-IDF features.
    cuisine_matrix = vectorizer.fit_transform(df["cuisine"].astype(str))
    cosine_sim = cosine_similarity(cuisine_matrix)  # Calculate cosine similarity between the TF-IDF vectors.

    # Get indices of rows where the 'cuisine' contains the user-specified cuisine.
    indices = df[df["cuisine"].str.contains(user_cuisine, case=False, na=False)].index.tolist()
    if not indices:
        st.error(f"No matches found for {user_cuisine}. Try another cuisine.")
        return []  # Return an empty list if no matching cuisine is found.

    scores = sorted(indices, key=lambda i: -cosine_sim[i].sum())[:top_n]  # Sort the indices based on the sum of similarity scores and select the top_n indices.
    # Return a subset of columns for the recommended restaurants.
    return df.iloc[scores][
        ["borough", "name", "restaurant_id", "cuisine", "halal", "rating", "user_id", "reviews", "place_id"]]


# Function to train a collaborative filtering model using Surprise's SVD.
def train_collaborative_model():
    df = load_restaurant_data()  # Load the restaurant dataset.
    # Check if the dataset is loaded and contains necessary columns.
    if df is None or not {"borough", "user_id", "restaurant_id", "rating"}.issubset(df.columns):
        return None

    reader = Reader(rating_scale=(1, 5))  # Create a Reader object with the specified rating scale.
    data = Dataset.load_from_df(df[["borough", "user_id", "restaurant_id", "rating"]], reader)  # Load the data from the DataFrame into a Surprise dataset.
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)  # Split the data into training and testing sets.

    model = SVD()  # Initialise the SVD model.
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
    # Analyses the sentiment of the review.
    # Returns the polarity of the review (positive, neutral, or negative).
    blob = TextBlob(review_text) # Create a TextBlob object with the review text.
    polarity = blob.sentiment.polarity # Extract the sentiment polarity from the TextBlob object.
    # Return a sentiment label based on the polarity value.
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def get_updated_rating(restaurant):

    # Calculate the average rating for a restaurant based on user ratings.
    # If no user ratings exist, return None.
    # Create a list of ratings for the restaurant from all users.
    ratings = [users[user]["ratings"].get(restaurant) for user in users if restaurant in users[user]["ratings"]]
    ratings = [r for r in ratings if r is not None]  # Remove any None values from the list.
    # If there are ratings available, compute the average and round it to 2 decimal places.
    if ratings:
        return round(sum(ratings) / len(ratings), 2)
    return None  # Return None if there are no ratings.


def submit_review_and_update_rating(user, restaurant, review_text, rating=None):

    # Handles user review submission and updates their rating.
    # If a rating is provided, it updates the user's rating for the restaurant.

    if review_text:  # If the user provided a review, save it under the restaurant key.
        users[user]["reviews"][restaurant] = review_text
    if rating is not None:  # If the user provided a rating, save it.
        users[user]["ratings"][restaurant] = rating
    save_users(users)  # Save the updated user data.
    return get_updated_rating(restaurant)  # Return the updated average rating for the restaurant.


def hybrid_recommend(user_id, user_cuisine, top_n=100):
    df = load_restaurant_data()  # Load the restaurant dataset.
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
        if pd.isna(restaurant_id) or pd.isna(place_id):  # Skip if restaurant_id or place_id is missing.
            continue
        try:
            pred = collaborative_model.predict(user_id, int(restaurant_id)).est  # Use the collaborative filtering model to predict a rating for this restaurant.
            google_reviews = get_google_reviews(place_id)  # Get Google reviews for the restaurant using its place_id.
            sentiments = [analyse_sentiment(review) for review in google_reviews]  # Analyse the sentiment of each Google review.
            predictions.append((restaurant_name, pred, google_reviews, sentiments))  # Append the restaurant name, predicted rating, reviews, and sentiments as a tuple.
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


# Session State Initialisation
# Initialise session state variables if not already set.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Reload users from file.
users = load_users()

# Sidebar Navigation
st.sidebar.title("🍽️ London Bites")
menu = st.sidebar.radio("Select an option:", ["Login", "Register"])  # Create a sidebar radio selection for "Login" and "Register".

# Login section
if menu == "Login":
    st.sidebar.subheader("🔑Login")
    username = st.sidebar.text_input("Username")  # Input field for username.
    password = st.sidebar.text_input("Password", type="password")  # Input field for password (hidden).
    login_button = st.sidebar.button("Login")  # Button to trigger login.
    if login_button:
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):  # Check if username exists and if the password matches using bcrypt.
            # Update session state variables to mark the user as authenticated.
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users[username]["user_id"]
            st.sidebar.success(f"✅Welcome, {users[username]['name']}!🎉")
        else:
            st.sidebar.error("❌Invalid username or password!")

# Register Section
elif menu == "Register":
    st.sidebar.subheader("📝 Create an Account")
    # Input fields for new user registration.
    new_username = st.sidebar.text_input("Choose a Username")
    new_name = st.sidebar.text_input("Full Name")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")
    if st.sidebar.button("Register"):
        users = load_users()  # Reload users to ensure the latest data.
        # Validate that all fields are provided.
        if not new_username or not new_name or not new_password or not confirm_password:
            st.sidebar.error("⚠️ All fields are required!")
        elif new_username in users:
            st.sidebar.error("❌ Username already exists! Choose a different one.")
        elif new_password != confirm_password:
            st.sidebar.error("⚠️ Passwords do not match!")
        else:
            try:
                user_id = len(users) + 1  # Generate a new user_id.
                hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()  # Hash the password.
                # Create a new user entry with all required fields.
                users[new_username] = {
                    "name": new_name,
                    "password": hashed_password,
                    "user_id": user_id,
                    "favourites": [],
                    "saved_recommendations": [],
                    "ratings": {},
                    "reviews": {},
                    "following": []  # Initialise following as an empty list.
                }
                # Save the updated user data.
                save_users(users)
                st.sidebar.success("🎉 Account created! Please log in.")
            except Exception as e:
                st.sidebar.error(f"Error creating account: {e}")

# Logout section
if st.session_state["authenticated"]:
    if st.sidebar.button("🚪Logout"):  # Provide a logout button if the user is authenticated.
        st.session_state.clear()  # Clear session state to log out.
        st.sidebar.success("✅Logged out!")

# Sidebar Menu Navigation for Home Page
menu = st.sidebar.radio("Go to", ["Home"], key="main_menu")

# Home page section
if st.session_state["authenticated"]:
    if menu == "Home":
        st.title("Find Your Next Meal 🍽️")
        if df is not None and "cuisine" in df.columns:  # If the dataset is loaded and has a 'cuisine' column, get unique cuisines.
            cuisines = sorted(df["cuisine"].dropna().unique())
        else:
            cuisines = []
        st.write("### Choose Your Cuisine:")
        # Provide a dropdown for selecting a cuisine.
        selected_cuisine = st.selectbox("Select Cuisine", cuisines, index=None, placeholder="Choose a cuisine")
        if selected_cuisine:
            st.write(f"### 🍽️ Restaurants serving {selected_cuisine}:")
            recommendations = hybrid_recommend(st.session_state["user_id"], selected_cuisine)  # Get hybrid recommendations for the selected cuisine.
            if not recommendations:
                st.warning("⚠️No restaurants found for this cuisine.")
            else:
                for restaurant, rating, google_reviews, sentiments in recommendations:  # Loop through each recommended restaurant.
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
