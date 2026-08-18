"""
El-Peak — P1.5 Load Testing Script (Locust)
Simulates ~200 concurrent users mixing: login -> static quiz -> AI quiz -> submit.
Run with: locust -f locustfile.py --host=https://your-railway-url.app
"""
from locust import HttpUser, task, between
import random

class ElPeakStudent(HttpUser):
    wait_time = between(1, 3)
    token = None
    user_email = None

    def on_start(self):
        """Register a new user or login if they already exist."""
        self.user_email = f"loadtest_{random.randint(10000, 99999)}@example.com"
        password = "password123"
        
        # Try to signup
        res = self.client.post("/api/auth/signup", json={
            "email": self.user_email,
            "password": password,
            "username": f"tester_{random.randint(1000, 9999)}",
            "language": "en",
            "user_type": "general",
            "grade": "g10-12"
        })
        
        if res.status_code != 200:
            # Fallback to login if user exists
            res = self.client.post("/api/auth/login", data={
                "username": self.user_email,
                "password": password
            })
        
        if res.status_code == 200:
            self.token = res.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})

    @task(3)
    def static_quiz_flow(self):
        """Fetch a static quiz and submit the results."""
        if not self.token: return
        
        res = self.client.get("/api/quiz/static?topic=Math&n=4&difficulty=mixed")
        if res.status_code == 200:
            data = res.json()
            session_id = data.get("session_id")
            self.client.post("/api/quiz/submit", json={
                "topic": "Math",
                "correct": 3,
                "total": 4,
                "source": "static",
                "session_id": session_id
            })

    @task(1)
    def ai_quiz_flow(self):
        """Generate an AI quiz (tests cache/single-flight) and submit."""
        if not self.token: return
        
        topics = ["Physics", "Biology", "History", "General Knowledge"]
        topic = random.choice(topics)
        
        res = self.client.post("/api/quiz/ai", json={
            "topic": topic,
            "num_questions": 4,
            "difficulty": "mixed"
        })
        
        if res.status_code == 200:
            data = res.json()
            session_id = data.get("session_id")
            self.client.post("/api/quiz/submit", json={
                "topic": topic,
                "correct": 2,
                "total": 4,
                "source": "ai",
                "session_id": session_id
            })
            
    @task(1)
    def check_dashboard(self):
        """Hit the dashboard stats endpoints."""
        if not self.token: return
        self.client.get("/api/stats")
        self.client.get("/api/topics")
