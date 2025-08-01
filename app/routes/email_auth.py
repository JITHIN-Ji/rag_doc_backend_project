from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models import chat_models as schemas
from app.models import question_store as db_access
from app import security

router = APIRouter()

@router.post("/register", response_model=schemas.UserInDB, tags=["Email Authentication"])
async def register_user(user: schemas.UserCreate):
    """
    Handles user registration with email and password.
    """
    db_user = await db_access.get_user_by_email(email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    hashed_password = security.get_password_hash(user.password)
    new_user = await db_access.create_user(email=user.email, hashed_password=hashed_password)
    return new_user


@router.post("/token", response_model=schemas.Token, tags=["Email Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Handles user login with email (as username) and password.
    Returns a JWT access token.
    """
    user = await db_access.get_user_by_email(email=form_data.username)
    if not user or not user.get("hashed_password") or not security.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    

    access_token = security.create_access_token(
        data={"sub": user["email"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}