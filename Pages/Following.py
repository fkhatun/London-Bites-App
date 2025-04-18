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
    with open(USER_DATA_FILE, "w") as file:  # Open the user data file (USER_DATA_FILE) in write mode. This will overwrite the existing file with the updated user data
        json.dump(users, file, indent=4)  # Convert the 'users' dictionary into JSON format and write it to the file
        # The 'indent=4' makes the JSON output nicely formatted and readable
    return load_users()  # Reload and return the updated dictionary.


# Function to add a review for a restaurant by a user.
def add_review(user, restaurant, review):
    users = load_users()  # Reload the latest user data.
    users[user]["reviews"][restaurant] = review  # Save the review for the restaurant under the given user.

    # Notify any followers of this user about the new review.
    for follower in users:
        if user in users[follower]["following"]:  # Check if the current user is in the follower's "following" list.
            users[follower]["notifications"].append(f"{user} reviewed {restaurant}: {review}")  # Append a notification message to the follower's notifications.

    # Save the updated user data.
    save_users(users)


# Function to add a rating for a restaurant by a user.
def add_rating(user, restaurant, rating):
    users = load_users()  # Reload the latest user data.
    users[user]["ratings"][restaurant] = rating  # Save the rating for the restaurant under the given user.

    # Notify followers about the new rating.
    for follower in users:
        if user in users[follower]["following"]:
            users[follower]["notifications"].append(f"{user} rated {restaurant} {rating} stars")

    # Save the updated user data.
    save_users(users)


# Session State Initialisation
# Initialise session state variables if they are not already set.
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Reload users data from file.
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


# Sidebar Menu Navigation for Following Page
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
selected_user = st.selectbox("Select a user to follow:", [""] + other_users)  # Provide a dropdown to select a user to follow.

# When the "Follow" button is pressed:
if st.button("Follow"):
    if selected_user and selected_user not in users[current_user]["following"]:  # Check if a user is selected and not already followed by the current user
        users[current_user]["following"].append(selected_user)  # Add user to the following list
        save_users(users)  # Save changes to user data
        st.success(f"You are now following {selected_user}!")  # Show confirmation message
    elif selected_user:
        st.warning(f"You are already following {selected_user}!")  # Warn if already following

# Always display followed users, along with an "Unfollow" option
st.subheader("Your Followed Friends")

if users[current_user]["following"]:  # Check if the user is following anyone
    for friend in users[current_user]["following"]:  # Loop through all followed friends
        # Create two columns: one for the friend's name and one for the Unfollow button.
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"👤 **{friend}**")  # Display the friend's username
        with col2:
            if st.button("Unfollow", key=f"unfollow_{friend}"):  # button to unfollow this user
                users[current_user]["following"].remove(friend)  # Remove from following list
                save_users(users)  # Save updated data
                st.experimental_rerun()  # Refresh the page to update the UI.

        # Display friend's favorite restaurants if available.
        if users[friend]["favourites"]:
            st.write("❤️ **Favorites:**")
            for fav in users[friend]["favourites"]:
                st.write(f"- {fav}")  # list each favourite

        # Display friend's ratings if available.
        if users[friend]["ratings"]:
            st.write("📊 **Ratings:**")
            for restaurant, rating in users[friend]["ratings"].items():
                st.write(f"- {restaurant}: ⭐ {rating}")  # show rating

        # Display friend's reviews if available.
        if users[friend]["reviews"]:
            st.write("📝 **Reviews:**")
            for restaurant, review in users[friend]["reviews"].items():
                st.write(f"- {restaurant}: {review}")  # show review contwnt
else:
    st.write("You are not following anyone yet.")  # if user is not following anyone

