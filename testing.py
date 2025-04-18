import unittest
import bcrypt
import json
import os
import tempfile

# Mock users data for testing
TEST_USERS = {
    "testuser": {
        "name": "Test User",
        "password": bcrypt.hashpw("password123".encode(), bcrypt.gensalt()).decode(),
        "user_id": 1,
        "favourites": [],
        "saved_recommendations": [],
        "ratings": {},
        "reviews": {},
        "following": []
    }
}


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for user data
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(self.temp_file.name, 'w') as f:
            json.dump(TEST_USERS, f)

        # Import functions from main app (would be imported from your actual app)
        self.load_users = lambda: json.load(open(self.temp_file.name, 'r'))
        self.save_users = lambda users: json.dump(users, open(self.temp_file.name, 'w'))

    def tearDown(self):
        # Remove temporary file
        os.unlink(self.temp_file.name)

    def test_password_hashing(self):
        """Test that password hashing produces different hashes for the same password"""
        password = "securepassword"
        hash1 = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        hash2 = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        # Verify that the same password generates different hashes due to salting
        self.assertNotEqual(hash1, hash2)

        # Verify both hashes validate against the original password
        self.assertTrue(bcrypt.checkpw(password.encode(), hash1))
        self.assertTrue(bcrypt.checkpw(password.encode(), hash2))

        print("Password hashing test passed - different salts generate unique hashes")

    def test_login_validation(self):
        """Test login validation with correct and incorrect credentials"""
        users = self.load_users()

        # Test with correct credentials
        valid_username = "testuser"
        valid_password = "password123"
        self.assertTrue(
            valid_username in users and
            bcrypt.checkpw(valid_password.encode(), users[valid_username]["password"].encode())
        )
        print("Valid login credentials test passed")

        # Test with incorrect password
        invalid_password = "wrongpassword"
        self.assertFalse(
            valid_username in users and
            bcrypt.checkpw(invalid_password.encode(), users[valid_username]["password"].encode())
        )
        print("Invalid password test passed")

        # Test with non-existent username
        invalid_username = "nonexistentuser"
        self.assertFalse(invalid_username in users)
        print("Invalid username test passed")

    def test_registration_password_storage(self):
        """Test that passwords are properly hashed before storage"""
        users = self.load_users()

        # Test adding a new user
        new_username = "newuser"
        new_password = "newpassword123"

        # Hash password as your registration code does
        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

        # Add new user to the mock database
        users[new_username] = {
            "name": "New User",
            "password": hashed_password,
            "user_id": 2,
            "favourites": [],
            "saved_recommendations": [],
            "ratings": {},
            "reviews": {},
            "following": []
        }
        self.save_users(users)

        # Reload users and verify
        users = self.load_users()

        # Check that password is not stored in plaintext
        self.assertNotEqual(users[new_username]["password"], new_password)

        # Verify the stored hash validates against the original password
        self.assertTrue(bcrypt.checkpw(new_password.encode(), users[new_username]["password"].encode()))
        print("Password storage test passed - passwords are properly hashed")


if __name__ == "__main__":
    unittest.main()