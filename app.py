import os
from time import time
import time
from typing import Literal
import sqlite3
import uuid
import json
import secrets
import bcrypt
import logging

import pandas as pd

import torch
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pydantic import BaseModel, Field, EmailStr
from transformers import AutoModelForCausalLM, AutoTokenizer

from database import get_db_connection
from download_image import download_meal_image

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOOKUP_CSV_FILE_PATH = "data/Food_and_Nutrition__.csv"
MODEL_CHECKPOINT = "syubraj/DietRecommender_4bit_Qwen2.5-0.5B"

torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lookup_df = pd.read_csv(LOOKUP_CSV_FILE_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT)
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

app = FastAPI(
    title="Diet Recommendation",
    description="Your personalized diet recommendations with macronutrient breakdown."
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

secret_key = secrets.token_hex(32) 
os.environ["SECRET_KEY"] = secret_key 


class DietPlanResponse(BaseModel):
    breakfast: str
    lunch: str
    snack: str
    dinner: str

class UserInput(BaseModel):
    age: int = Field(..., ge=1, le=120, title="User's Age")
    height: float = Field(..., title="Height of the user in cm")
    weight: float = Field(..., title="Weight of the user in kg")
    gender: Literal['male', 'female'] = Field(..., title="Gender of user")
    diet_preference: Literal["Vegetarian", "Vegan", "Pescatarian", "Omnivorous"] = Field(..., title="User's dietary preference")
    activity_level: Literal['Sedentry', 'Lightly Active','Moderately Active', 'Very Active'] = Field(..., title="User's daily activity level")
    daily_calorie_target: int = Field(..., title="Target daily calorie intake in kcal")

class LogMeal(BaseModel):
    meal_type: Literal['breakfast', 'lunch', 'snack', 'dinner'] = Field(..., title="Part of the meal user had.")
    recommended_diet: str

class RegisterUser(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    age: int
    weight: float
    gender: Literal["male", "female"]
    height: int 

class UserResponse(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    age: int
    weight: float
    gender: Literal["male", "female"]
    height: int

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def generate_diet_prompt(user_request: UserInput) -> str:
    """
    Generates a structured prompt for an AI nutrition expert to suggest a balanced meal plan.
    """
    
    prompt = f"""<|im_start|>system
        Act as a nutrition expert. Based on the user’s age, gender, height, weight, activity level, diet preference, and calorie target, 
        suggest a balanced meal plan for a day only with breakfast, lunch, snacks, and dinner.Do consider users dietary preference and suggest me atleast 3 suggestions.<|im_end|>

        <|im_start|>user
        Age: {user_request.age}
        Gender: {user_request.gender}
        Height: {user_request.height} cm
        Weight: {user_request.weight} kg
        Activity Level: {user_request.activity_level}
        **Dietary Preference: {user_request.diet_preference}**
        Daily Calorie Target: {user_request.daily_calorie_target} kcal<|im_end|>

        <|im_start|>assistant
        """
    return prompt


def diet_via_time(tm: int):
    """Function to return meal type based on time of the day."""
    if 5 <= tm < 10:
        return 'breakfast'
    elif 10 <= tm < 14:
        return 'lunch'
    elif 14 <= tm < 18:
        return 'snack'
    elif 18 <= tm <= 24:
        return 'dinner'
    else:
        return None

def get_macronutrients(
    meal_type: str,
    recommended_diet: str,
    lookup_df: pd.DataFrame = lookup_df
):
    """
    Look up macronutrients for the predicted diet from the lookup table.
    """
    if lookup_df is None:
        raise ValueError("Lookup DataFrame is required.")

    cols_to_return = ["Protein", "Fat", "Fiber", "Carbohydrates", "Calories"]

    column_name = f"{meal_type.capitalize()} Suggestion"

    if column_name not in lookup_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the lookup DataFrame.")

    diet_info = lookup_df[lookup_df[column_name] == recommended_diet]

    if not diet_info.empty:
        logger.debug(f"Macronutrients found for {meal_type}: {diet_info[cols_to_return].to_dict(orient='records')[0]}")
        return diet_info[cols_to_return].to_dict(orient="records")[0]  
    else:
        return None    

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
def login(
    email: EmailStr = Form(...), 
    password: str = Form(...),    
    db: sqlite3.Connection = Depends(get_db),
):
    try:
        cursor = db.cursor()
        cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if not user:
            logger.error("Invalid email or password")
            return JSONResponse(content={"error": "Invalid email or password"}, status_code=400)

        user_id, hashed_password = user

        if not verify_password(password, hashed_password):
            logger.error("Invalid email or password")
            return JSONResponse(content={"error": "Invalid email or password"}, status_code=400)
        
        response = RedirectResponse(url='/dashboard', status_code=303)
        response.set_cookie(key="user_id", value=user_id, max_age=600) 
        logger.info("User logged in successfully")
        return response        

    except Exception as e:
        logger.error(f"An error occurred during login: {e}")
        return JSONResponse(content={"error": "An error occurred during login. Please try again."}, status_code=500)

@app.get("/register/")
def show_register_form(request: Request):
    logger.info("Showing registration form")
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register/")
def register(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    if password != confirm_password:
        logger,error("Passwords do not match!")
        return {"error": "Passwords do not match!"}

    hashed_password = hash_password(password)
    user_id = str(uuid.uuid4())

    try:
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (id, full_name, email, password, age, weight, gender, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, full_name, email, hashed_password, age, weight, gender, height)
        )
        db.commit()
    except sqlite3.IntegrityError as e:
        logger.error(f"Email already registered: {e}")
        raise HTTPException(status_code=400, detail="Email already registered.")
    except Exception as e:
        logger.error(f"An error occurred while registering the user: {e}")
        raise HTTPException(status_code=500, detail=f"An error: {e} occurred while registering the user.")

    return RedirectResponse(url="/", status_code=303)

    
@app.get("/suggest-meal/")
def get_recommendation_template(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        logger.info("User logged in. Showing diet recommendation page.")
        return templates.TemplateResponse("recommendation.html", {"request": request})
    else:
        logger.warning("Unauthorized access attempt - no user_id found in cookies")
        return RedirectResponse(url="/", status_code=303)

@app.post("/suggest-meal/")
def get_recommendation(
    request: Request,
    daily_calorie_target: int = Form(...),
    activity_level: str = Form(...),
    diet_preference: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    try:
        logger.info("Received diet recommendation request")
        
        user_id = request.cookies.get("user_id")
        if not user_id:
            logger.warning("Unauthorized access attempt - no user_id found in cookies")
            raise HTTPException(status_code=401, detail="User not logged in")

        logger.debug(f"Fetching user details for user_id: {user_id}")

        cursor = db.cursor()
        cursor.execute("SELECT age, weight, height, gender FROM users WHERE id = ?", (user_id.strip(),))
        user = cursor.fetchone()

        if not user:
            logger.error(f"No user found in database for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")

        age, weight, height, gender = user["age"], user["weight"], user["height"], user["gender"]
        logger.debug(f"User details: Age={age}, Weight={weight}, Height={height}, Gender={gender}")

        user_request = UserInput(
            age=age,
            weight=weight,
            height=height,
            gender=gender,
            daily_calorie_target=daily_calorie_target,
            activity_level=activity_level,
            diet_preference=diet_preference
        )

        current_hour = time.localtime().tm_hour
        current_minute = time.localtime().tm_min
        meal_type = diet_via_time(tm=current_hour)
        logger.debug(f"Meal type determined: {meal_type}")

        logger.debug("Generating diet prompt...")
        prompt = generate_diet_prompt(user_request=user_request)

        try:
            logger.debug("Tokenizing prompt...")
            input_data = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        except Exception as e:
            logger.error(f"Tokenizer error: {e}")
            raise HTTPException(status_code=500, detail=f"Tokenizer error: {e}")

        try:
            logger.debug("Generating diet recommendation from model...")
            output_ids = model.generate(
                input_data["input_ids"], 
                attention_mask=input_data["attention_mask"], 
                max_length=1000,
                temperature=0.7, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id 
            )
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation error: {e}")

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        lines = response.strip().split("\n")
        parsed_data = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_data[key.strip().lower()] = value.strip()

        parsed_data = DietPlanResponse(**parsed_data).model_dump()
        recommended_meal = parsed_data.get(meal_type, "")

        image_path = download_meal_image(meal_name=recommended_meal)
        logger.info(f"Meal recommendation generated successfully: {recommended_meal}")

        return {
            "current_time": f"{current_hour}:{current_minute}",
            "image_path": f"../{image_path}",
            "meal_type": meal_type,
            "recommended_diet": recommended_meal
        }

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}") from e

@app.post("/log-meal/")
def log_your_meal(
    logmeal: LogMeal, 
    request: Request,  
    db: sqlite3.Connection = Depends(get_db)
):
    try:
        user_id = request.cookies.get("user_id")
        if not user_id:
            logger.error("Unauthorized access attempt - no user_id found in cookies")
            raise HTTPException(status_code=401, detail="User not logged in")

        meal_type = logmeal.meal_type
        recommended_diet = logmeal.recommended_diet
        current_date = time.strftime("%Y-%m-%d")
        macro_nutrients = get_macronutrients(meal_type=meal_type, recommended_diet=recommended_diet)

        if macro_nutrients:
            cursor = db.cursor()
            cursor.execute(
                """INSERT INTO macronutrients 
                (user_id, meal_type, recommended_diet, calories_intake, protein, carbohydrates, fats, fibre, date) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    meal_type,
                    recommended_diet,
                    macro_nutrients["Calories"],
                    macro_nutrients["Protein"],
                    macro_nutrients["Carbohydrates"],
                    macro_nutrients["Fat"],
                    macro_nutrients["Fiber"],
                    current_date 
                )
            )
            db.commit()
        logger.info("Meal logged successfully")
        return JSONResponse(content={"message": "Meal logged successfully"}, status_code=200)

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/dashboard/")
def dashboard_page(request: Request, db: sqlite3.Connection = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        logger.error("Unauthorized access attempt - no user_id found in cookies")
        return RedirectResponse(url="/")

    cursor = db.cursor()

    cursor.execute(
        "SELECT meal_type, recommended_diet, calories_intake, protein, carbohydrates, fats, fibre, date "
        "FROM macronutrients WHERE user_id=?",
        (user_id,)
    )

    meals = cursor.fetchall()

    meal_data = [
        {
            "meal_type": row["meal_type"],
            "recommended_diet": row["recommended_diet"],
            "calories": row["calories_intake"],
            "protein": row["protein"],
            "carbohydrates": row["carbohydrates"],
            "fats": row["fats"],
            "fiber": row["fibre"],
            "date": row["date"],
        }
        for row in meals
    ]

    cursor.execute(
        "SELECT SUM(calories_intake) AS total_calories, SUM(protein) AS total_protein, "
        "SUM(carbohydrates) AS total_carbohydrates, SUM(fats) AS total_fats, SUM(fibre) AS total_fiber "
        "FROM macronutrients WHERE user_id=?",
        (user_id,)
    )
    total_nutrition = cursor.fetchone()

    summary = {
        "total_calories": total_nutrition["total_calories"] or 0,
        "total_protein": total_nutrition["total_protein"] or 0,
        "total_carbohydrates": total_nutrition["total_carbohydrates"] or 0,
        "total_fats": total_nutrition["total_fats"] or 0,
        "total_fiber": total_nutrition["total_fiber"] or 0,
    }

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        logger.info("Returning JSON response for AJAX")
        return JSONResponse({"meals": meal_data, "summary": summary})

    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "nutrition_data": json.dumps(meal_data)}
    )


@app.get("/logout/")
def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="user_id")
    logger.info("User logged out successfully")  
    return response