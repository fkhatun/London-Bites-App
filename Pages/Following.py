import json
import os
import bcrypt
import streamlit as st

# Define the path to the JSON file where user data is stored.
USER_DATA_FILE = "users.json"


# Function to load user data from the JSON file.
def load_users():
    # Check if the user data file exists.
    if os.path.exists(USER_DATA_FILE):
        # Open and load the JSON file into a Python dictionary.
        with open(USER_DATA_FILE, "r") as file:
            users = json.load(file)
    else:
        # If the file doesn't exist, initialize an empty dictionary.
        users = {}

    # For each user in the dictionary, ensure that the "following" key exists and is a list.
    for user in users:
        users[user].setdefault("following", [])  # This ensures each user has a "following" list.
    return users


# Function to save user data to the JSON file.
def save_users(users):
    # Open the file in write mode and dump the users dictionary into it with indentation.
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)


# Load users from file into the variable 'users'.
users = load_users()


# Function to add a review for a restaurant by a user.
def add_review(user, restaurant, review):
    # Reload the latest users data.
    users = load_users()
    # Save the review for the restaurant under the given user.
    users[user]["reviews"][restaurant] = review

    # Notify any followers of this user about the new review.
    for follower in users:
        # Check if the current user is in the follower's "following" list.
        if user in users[follower]["following"]:
            # Append a notification message to the follower's notifications.
            users[follower]["notifications"].append(f"{user} reviewed {restaurant}: {review}")

    # Save the updated user data.
    save_users(users)


# Function to add a rating for a restaurant by a user.
def add_rating(user, restaurant, rating):
    # Reload the latest users data.
    users = load_users()
    # Save the rating for the restaurant under the given user.
    users[user]["ratings"][restaurant] = rating

    # Notify followers about the new rating.
    for follower in users:
        if user in users[follower]["following"]:
            users[follower]["notifications"].append(f"{user} rated {restaurant} {rating} stars")

    # Save the updated user data.
    save_users(users)


# ---- Session State Initialization ----
# Initialize session state variables if they are not already set.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Reload users data from file.
users = load_users()


# ---- Sidebar Navigation ----
# Set the title for the sidebar.
st.sidebar.title("🍽️ London Bites")
# Create a radio button selection for "Login" and "Register" options.
menu = st.sidebar.radio("Select an option:", ["Login", "Register"])

# ---- LOGIN SECTION ----
if menu == "Login":
    st.sidebar.subheader("🔑Login")
    # Input field for the username.
    username = st.sidebar.text_input("Username")
    # Input field for the password (masked).
    password = st.sidebar.text_input("Password", type="password")
    # Create a button to trigger the login.
    login_button = st.sidebar.button("Login")

    # If the login button is pressed:
    if login_button:
        # Check if the username exists and the provided password matches (using bcrypt to verify).
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            # Set session state variables to mark the user as authenticated.
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = users[username]["user_id"]
            # Display a success message.
            st.sidebar.success(f"✅Welcome, {users[username]['name']}!🎉")
        else:
            # Display an error message if login fails.
            st.sidebar.error("❌Invalid username or password!")

# ---- REGISTER SECTION ----
elif menu == "📝Register":
    st.sidebar.subheader("Create an Account")
    # Input fields for new user registration.
    new_username = st.sidebar.text_input("Choose a Username")
    new_name = st.sidebar.text_input("Full Name")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")

    # When the Register button is pressed:
    if st.sidebar.button("Register"):
        # Check if the username already exists.
        if new_username in users:
            st.sidebar.error("Username already exists!")
        # Check if the two passwords match.
        elif new_password != confirm_password:
            st.sidebar.error("⚠️Passwords do not match!")
        else:
            # Generate a new user_id based on the current number of users.
            user_id = len(users) + 1
            # Hash the new password using bcrypt.
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

            # Create a new user dictionary with all the required fields.
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
            # Save the new user data.
            save_users(users)
            st.sidebar.success("🎉Account created! Please log in.")

# ---- LOGOUT SECTION ----
# If the user is authenticated, display a logout button.
if st.session_state["authenticated"]:
    if st.sidebar.button("🚪Logout"):
        # Clear the session state to log the user out.
        st.session_state.clear()
        st.sidebar.success("✅Logged out!")

# ---- Sidebar Menu Navigation for Following Page ----
# Create a new sidebar radio option to navigate to the "Following" page.
menu = st.sidebar.radio("Go to", ["Following"], key="main_menu")

# Title for the Follow Friends page.
st.title("👥 Follow Friends")

# Ensure that the user is logged in; if not, display a warning and stop execution.
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("You must be logged in to follow friends.")
    st.stop()

# Get the current username from session state.
current_user = st.session_state["username"]

# Create a list of other users (exclude the current user).
other_users = [user for user in users.keys() if user != current_user]
# Provide a dropdown to select a user to follow.
selected_user = st.selectbox("Select a user to follow:", [""] + other_users)

# When the "Follow" button is pressed:
if st.button("Follow"):
    if selected_user and selected_user not in users[current_user]["following"]:
        users[current_user]["following"].append(selected_user)
        save_users(users)
        st.success(f"You are now following {selected_user}!")
    elif selected_user:
        st.warning(f"You are already following {selected_user}!")

# ---- ALWAYS SHOW FOLLOWED USERS WITH UNFOLLOW BUTTON ----
st.subheader("Your Followed Friends")
if users[current_user]["following"]:
    for friend in users[current_user]["following"]:
        # Create two columns: one for the friend's name and one for the Unfollow button.
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"👤 **{friend}**")
        with col2:
            if st.button("Unfollow", key=f"unfollow_{friend}"):
                users[current_user]["following"].remove(friend)
                save_users(users)
                st.experimental_rerun()  # Refresh the page to update the UI.

        # Display friend's favorite restaurants if available.
        if users[friend]["favourites"]:
            st.write("❤️ **Favorites:**")
            for fav in users[friend]["favourites"]:
                st.write(f"- {fav}")

        # Display friend's ratings if available.
        if users[friend]["ratings"]:
            st.write("📊 **Ratings:**")
            for restaurant, rating in users[friend]["ratings"].items():
                st.write(f"- {restaurant}: ⭐ {rating}")

        # Display friend's reviews if available.
        if users[friend]["reviews"]:
            st.write("📝 **Reviews:**")
            for restaurant, review in users[friend]["reviews"].items():
                st.write(f"- {restaurant}: {review}")
else:
    st.write("You are not following anyone yet.")

