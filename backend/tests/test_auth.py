from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup_and_login(monkeypatch):
    email = "test.user@example.com"
    password = "a-strong-password"

    signup_response = client.post("/auth/signup", json={"email": email, "password": password})
    assert signup_response.status_code == 201
    body = signup_response.json()
    assert body["onboarding_completed"] is False
    assert "access_token" in body

    login_response = client.post("/auth/login", json={"email": email, "password": password})
    assert login_response.status_code == 200
