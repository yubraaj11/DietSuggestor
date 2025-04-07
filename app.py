import os
from time import time
import time
from typing import Literal, List, Dict
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
from utils import vegetarian_filter

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_CHECKPOINT = "Raasmii/FitFoodie_4bit_Qwen2.5-0.5B"

torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

class RecipeRequest(BaseModel):
    ingredients: List[str]

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


def generate_diet_prompt(user_request: UserInput, tokenizer = tokenizer) -> str:
    """
    Generates a structured prompt for an AI nutrition expert to suggest multiple balanced meal plans,
    ensuring strict adherence to the user's dietary preference and returning only the required JSON.
    """
    
    system_prompt = """Act as a nutrition expert. Based on the user’s age, gender, height, weight, activity level, diet preference, and calorie target, 
suggest at least three different balanced meal plans for a day. Each meal plan should include breakfast, lunch, snacks, and dinner.

**Important Guidelines:**
- Only suggest meals that strictly follow the user's dietary preference. 
- If the user is vegetarian, exclude all meat, poultry, fish, and seafood. 
- If the user is vegan, exclude all animal products, including dairy and eggs.
- If the user follows any other specific diet, adhere strictly to its rules.
- Ensure that the total calorie count aligns with the daily calorie target.

**Example response format:**
```json
{
    "meal_plans": [
        {
            "breakfast": "Example breakfast",
            "lunch": "Example lunch",
            "snack": "Example snack",
            "dinner": "Example dinner"
        },
        {
            "breakfast": "Example breakfast 2",
            "lunch": "Example lunch 2",
            "snack": "Example snack 2",
            "dinner": "Example dinner 2"
        },
                {
            "breakfast": "Example breakfast 3",
            "lunch": "Example lunch 3",
            "snack": "Example snack 3",
            "dinner": "Example dinner 3"
        },
    ]
}
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Age: {user_request.age}
                                        Gender: {user_request.gender}
                                        Height: {user_request.height} cm
                                        Weight: {user_request.weight} kg
                                        Activity Level: {user_request.activity_level}
                                        Dietary Preference: {user_request.diet_preference}
                                        Daily Calorie Target: {user_request.daily_calorie_target} kcal"""},
                                            ]

    diet_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return diet_prompt


def generate_recipe_prompt(ingredients):
    """Generates a chat template prompt for recipe generation with JSON response format."""
    
    system_prompt = """
        Act as a chef. Create a dish name and provide clear, detailed step-by-step instructions based on the given ingredients. 
        Use only the listed ingredients as core components, but assume basic pantry items (e.g., salt, pepper, oil) if needed. 
        Respond in JSON format with 'dish_name' and 'steps'.

    **Example Response Format:**
    {
        "dish_name": "<dish name>",
        "steps": [
            "<step 1>",
            "<step 2>",
            "<step 3>",
            ...
        ]
    }
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Ingredients: {', '.join(ingredients)}"}
    ]

    recipe_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return recipe_prompt

def generate_macro_prompt(meal):
    """Generates a chat template prompt for macronutrient estimation with JSON response format."""
    
    system_prompt = """
        Act as a nutritionist. Based on the given meal, estimate the total macronutrient breakdown. 
        Provide values for Calories in kcal, and remaining Protein, Carbohydrates, Fat, and Fiber in grams. 
        Respond in JSON format with the following keys: 'Calories', 'Protein', 'Carbohydrates', 'Fat', and 'Fiber'.

    **Example Response Format:**
    {
        "Calories": <total_calories_in_kcal>,
        "Protein": <total_protein_in_grams>,
        "Carbohydrates": <total_carbohydrates_in_grams>,
        "Fat": <total_fat_in_grams>,
        "Fiber": <total_fiber_in_grams>
    }
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Meal: {meal}"}
    ]

    macro_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return macro_prompt

def diet_via_time(tm: int):
    """Function to return meal type based on time of the day."""
    if 0 <= tm < 10:
        return 'breakfast'
    elif 10 <= tm < 14:
        return 'lunch'
    elif 14 <= tm < 18:
        return 'snack'
    elif 18 <= tm <= 24:
        return 'dinner'
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
        response.set_cookie(key="user_id", value=user_id, max_age=1200) 
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
        logger.error("Passwords do not match!")
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
        logger.debug(f"Diet preference: {diet_preference}")
        
        user_id = request.cookies.get("user_id")
        if not user_id:
            logger.warning("Unauthorized access attempt - no user_id found in cookies")
            return RedirectResponse(url="/", status_code=303)

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
            input_data = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True).to(device=DEVICE)
        except Exception as e:
            logger.error(f"Tokenizer error: {e}")
            raise HTTPException(status_code=500, detail=f"Tokenizer error: {e}")

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_data["input_ids"],
                    attention_mask=input_data["attention_mask"],
                    max_new_tokens=1000,  
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1, 
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  
                )
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation error: {e}")

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(response)
        
        response = response.split("assistant")[1].strip()

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()

            meal_plans = json.loads(response)["meal_plans"]
            logger.debug(f"Meal plans extracted: {meal_plans}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response content: {response}")
            raise HTTPException(status_code=500, detail="Failed to parse JSON response. Ensure the model outputs valid JSON.")
        
        except KeyError as e:
            logger.error(f"Key 'meal_plans' not found in JSON response: {e}")
            logger.error(f"Raw response content: {response}")
            raise HTTPException(status_code=500, detail="Invalid JSON format. 'meal_plans' key is missing.")

        meal_options = [plan[meal_type] for plan in meal_plans if meal_type in plan]
        if not meal_options:
            logger.error(f"No meal options found for type: {meal_type}")
            raise HTTPException(status_code=500, detail=f"No meal options found for type: {meal_type}")

        if diet_preference == "Vegetarian":
            meal_options = vegetarian_filter(meal_options)
            logger.debug(f"Applied vegetarian filter, resulting options: {meal_options}")

        meal_images = []
        for meal in meal_options:
            try:
                image_path = download_meal_image(meal_name=meal)
                meal_images.append({
                    "meal_name": meal,
                    "image_path": f"../{image_path}"
                })
            except Exception as e:
                logger.error(f"Failed to download image for meal {meal}: {e}")
                meal_images.append({
                    "meal_name": meal,
                    "image_path": "../static/images/default_meal.jpg"  # Use a default image path
                })

        logger.info(f"Meal recommendations generated successfully for {meal_type}")

        return {
            "current_time": f"{current_hour}:{current_minute}",
            "meal_type": meal_type,
            "meal_options": meal_images  
        }

    except HTTPException as e:
        raise e  
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/log-meal/")
def log_your_meal(
    logmeal: LogMeal,
    request: Request,
    db: sqlite3.Connection = Depends(get_db)
):
    logger.info(logmeal)
    try:
        user_id = request.cookies.get("user_id")
        if not user_id:
            logger.error("Unauthorized access attempt - no user_id found in cookies")
            raise HTTPException(status_code=401, detail="User not logged in")

        meal_type = logmeal.meal_type
        recommended_diet = logmeal.recommended_diet
        current_date = time.strftime("%Y-%m-%d")
        macro_prompt = generate_macro_prompt(meal=recommended_diet)
        try:
            logger.debug("Tokenizing prompt...")
            input_data = tokenizer(macro_prompt, return_tensors="pt", padding="longest", truncation=True).to(device=DEVICE)
        except Exception as e:
            logger.error(f"Tokenizer error: {e}")
            raise HTTPException(status_code=500, detail=f"Tokenizer error: {e}")

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_data["input_ids"],
                    attention_mask=input_data["attention_mask"],
                    max_new_tokens=300,  
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1, 
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  
                )
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Model generation error: {e}")

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(response)
        
        response = response.split("assistant")[1].strip()

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()

            macro_nutrients = json.loads(response)
            logger.debug(f"Meal plans extracted: {macro_nutrients}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response content: {response}")
            raise HTTPException(status_code=500, detail="Failed to parse JSON response. Ensure the model outputs valid JSON.")
        
        except KeyError as e:
            logger.error(f"Key 'meal_plans' not found in JSON response: {e}")
            logger.error(f"Raw response content: {response}")
            raise HTTPException(status_code=500, detail="Invalid JSON format. 'meal_plans' key is missing.")

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



@app.get("/craft-recipe/")
def craft_recipe_template(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        logger.info("User logged in. Showing recipe page.")
        return templates.TemplateResponse("craft_recipe.html", {"request": request})
    else:
        logger.warning("Unauthorized access attempt - no user_id found in cookies")
        return RedirectResponse(url="/", status_code=303)

@app.post("/craft-recipe/")
async def craft_recipe(
    request: Request,
    recipe_request: RecipeRequest,
    db: sqlite3.Connection = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    if not user_id:
        logger.warning("Unauthorized access attempt - no user_id found in cookies")
        return RedirectResponse(url="/", status_code=303)
    
    try:
        
        recipe_prompt = generate_recipe_prompt(ingredients=recipe_request.ingredients)
        
        logger.debug(f"Generating recipe for ingredients: {recipe_request.ingredients}")
        
        try:
            logger.debug("Tokenizing prompt...")
            input_data = tokenizer(
                recipe_prompt, 
                return_tensors="pt", 
                padding="longest", 
                truncation=True
            ).to(device=DEVICE)
        except Exception as e:
            logger.error(f"Tokenizer error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process ingredients: {str(e)}"
            )
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_data["input_ids"],
                    attention_mask=input_data["attention_mask"],
                    max_new_tokens=1000,  
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1, 
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True  
                )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA out of memory error: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Server is currently overloaded. Please try again later."
                )
            else:
                logger.error(f"Model generation error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate recipe. Please try again."
                )
        except Exception as e:
            logger.error(f"Unexpected error during model generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred. Please try again."
            )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.debug(f"Raw model response: {response}")
        
        try:
            if "assistant" in response:
                response = response.split("assistant")[1].strip()
            
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response: 
                response = response.split("```")[1].split("```")[0].strip()
            
            recipe = json.loads(response)
            logger.info(f"Parsed recipe: {recipe}")
            
            if "dish_name" not in recipe or "steps" not in recipe:
                logger.warning(f"Incomplete recipe generated: {recipe}")
                raise HTTPException(
                    status_code=422,
                    detail="Generated recipe is incomplete. Please try again with different ingredients."
                )

            return {
                "dish_name": recipe.get("dish_name"), 
                "steps": recipe.get("steps"),
                "ingredients": recipe_request.ingredients
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}, Response: {response}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse the generated recipe. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in craft_recipe: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/logout/")
def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="user_id")
    logger.info("User logged out successfully")  
    return response