import unittest
import time
from fastapi.testclient import TestClient
from api.main import app

class TestAuthRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_signup_and_login_flow(self):
        email = f"test_{int(time.time())}@governance.ai"
        
        # 1. Sign Up
        signup_payload = {
            "name": "Test User",
            "email": email,
            "password": "testpassword",
            "plan": "Standard Workspace"
        }
        res = self.client.post("/api/auth/signup", json=signup_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("token", data)
        self.assertIn("session", data)
        self.assertEqual(data["session"]["user"]["email"], email)
        self.assertEqual(data["session"]["user"]["name"], "Test User")
        self.assertEqual(data["session"]["user"]["plan"], "Standard Workspace")
        
        token = data["token"]
        
        # 2. Duplicate Sign Up should fail
        res_dup = self.client.post("/api/auth/signup", json=signup_payload)
        self.assertEqual(res_dup.status_code, 400)
        self.assertIn("already exists", res_dup.json()["detail"])
        
        # 3. Session Verification
        res_sess = self.client.get("/api/auth/session", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res_sess.status_code, 200)
        self.assertEqual(res_sess.json()["session"]["user"]["email"], email)
        
        # 4. Login with credentials
        login_payload = {
            "email": email,
            "password": "testpassword"
        }
        res_login = self.client.post("/api/auth/login", json=login_payload)
        self.assertEqual(res_login.status_code, 200)
        self.assertIn("token", res_login.json())
        
        # 5. Invalid login
        invalid_login = {
            "email": email,
            "password": "wrongpassword"
        }
        res_invalid = self.client.post("/api/auth/login", json=invalid_login)
        self.assertEqual(res_invalid.status_code, 401)
        
        # 6. Logout
        res_logout = self.client.post("/api/auth/logout")
        self.assertEqual(res_logout.status_code, 200)

    def test_oauth_login_redirect_google(self):
        res = self.client.get("/api/auth/oauth/login/google", follow_redirects=False)
        self.assertEqual(res.status_code, 307)
        self.assertIn("Location", res.headers)
        
    def test_oauth_login_redirect_github(self):
        res = self.client.get("/api/auth/oauth/login/github", follow_redirects=False)
        self.assertEqual(res.status_code, 307)
        self.assertIn("Location", res.headers)

    def test_oauth_callback_mock_google(self):
        res = self.client.get("/api/auth/oauth/callback/google?code=mock_google_code&state=mock_state", follow_redirects=False)
        self.assertEqual(res.status_code, 307)
        location = res.headers["Location"]
        self.assertIn("/?token=", location)

    def test_oauth_callback_mock_github(self):
        res = self.client.get("/api/auth/oauth/callback/github?code=mock_github_code&state=mock_state", follow_redirects=False)
        self.assertEqual(res.status_code, 307)
        location = res.headers["Location"]
        self.assertIn("/?token=", location)

if __name__ == '__main__':
    unittest.main()
