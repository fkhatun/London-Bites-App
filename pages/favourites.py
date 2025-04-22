import json  # working with user data
import os  # interacting with the file system
import bcrypt  # securely hashing and verifying user passwords
import pandas as pd  # data manipulation and analysis
import streamlit as st  # building London Bites app

# Streamlit App Config
st.set_page_config(page_title="London Bites", page_icon="🍽️")

# Defining file paths for user data, collaborative filtering model, and dataset.
USER_DATA_FILE = "users.json"
COLLABORATIVE_MODEL_FILE = "collaborative_model.pkl"
DATASET_PATH = "London dataset.csv"


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
    with open(USER_DATA_FILE,
              "w") as file:  # Open the user data file (USER_DATA_FILE) in write mode. This will overwrite the existing file with the updated user data
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


df = load_restaurant_data()


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
menu = st.sidebar.radio("Select an option:",
                        ["Login", "Register"])  # Create a sidebar radio selection for "Login" and "Register".

# Login section
if menu == "Login":
    st.sidebar.subheader("🔑Login")
    username = st.sidebar.text_input("Username")  # Input field for username.
    password = st.sidebar.text_input("Password", type="password")  # Input field for password (hidden).
    login_button = st.sidebar.button("Login")  # Button to trigger login.
    if login_button:
        if username in users and bcrypt.checkpw(password.encode(), users[username][
            "password"].encode()):  # Check if username exists and if the password matches using bcrypt.
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

# Sidebar Menu Navigation
menu = st.sidebar.radio("Go to", ["Favourites"], key="main_menu")

# Favourites Page
if st.session_state.get("authenticated") and menu == "Favourites":
    st.title("❤️ My Favourite Restaurants")  # title of the favourites page

    username = st.session_state["username"]
    user_favourites = users[username].get("favourites", [])  # Get the current user's favourites

    # Reload the latest user data from the JSON file to ensure it's updated
    users = load_users()
    favourites = users.get(username, {}).get("favourites", [])

    # If no favourites exist, show a message
    if not user_favourites:
        st.write("You have no favourite restaurants yet.")
    else:
        # Iterate through each favourite restaurant
        for fav in user_favourites:
            fav_data = df[df["name"] == fav]  # Filter the DataFrame for the current restaurant

            if fav_data.empty:
                continue  # Skip if the restaurant data is not found

            # Extract relevant details, using fallbacks if missing
            rating = fav_data["rating"].values[0] if "rating" in fav_data.columns else "N/A"
            borough = fav_data["borough"].values[0] if "borough" in fav_data.columns else "N/A"
            halal = fav_data["halal"].values[0] if "halal" in fav_data.columns else None

            # Display restaurant details
            st.write(f"### {fav}")
            st.write(f"📍 **Location**: {borough}")
            st.write(f"⭐ **Rating**: {rating}")
            st.write(f"حَلَال **Halal**: {halal}")

            # Remove from favourites button
            if st.button(f"Remove {fav} from Favourites", key=f"remove_{fav}"):
                users[username]["favourites"].remove(fav)  # Remove from list
                save_users(users)  # Save updated user data
                st.success(f"{fav} has been removed from your favourites!")  # Notify the user
                st.rerun()  # Refresh the page to reflect changes

