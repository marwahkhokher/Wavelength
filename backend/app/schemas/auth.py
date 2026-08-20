import uuid

from pydantic import BaseModel, EmailStr

from app.models.enums import Gender


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    gender: Gender | None = None  # FR-ONB-5 can also be captured at signup


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    onboarding_completed: bool  # tells the client whether to route to
    # the questionnaire (FR-AUTH-3) or Home (FR-AUTH-2)


class RefreshRequest(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    gender: Gender | None
    onboarding_completed: bool

    class Config:
        from_attributes = True
