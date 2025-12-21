from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv('D:/STUDY/UOL2025/CV_Course/nutrition_flask_app/static/daily_food_nutrition_dataset.csv', on_bad_lines='skip')
df.fillna(0, inplace=True)

# -------------------------------
# Encode Categorical Columns
# -------------------------------
le_food = LabelEncoder()
le_category = LabelEncoder()

df['Food_Item_encoded'] = le_food.fit_transform(df['Food_Item'])
df['Category_encoded'] = le_category.fit_transform(df['Category'].astype(str))

# -------------------------------
# Features & Target
# -------------------------------
X = df[['Food_Item_encoded', 'Category_encoded']]
y = df[
    [
        'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)',
        'Fiber (g)', 'Sugars (g)', 'Sodium (mg)',
        'Cholesterol (mg)', 'Water_Intake (ml)'
    ]
]

# -------------------------------
# Train Model
# -------------------------------
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42)
model = MultiOutputRegressor(xgb)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    food_items = sorted(df['Food_Item'].unique().tolist())
    categories = sorted(df['Category'].unique().tolist())
    prediction_text = ""
    
    if request.method == "POST":
        food_item = request.form.get("food_item")
        category = request.form.get("category")
        try:
            food_encoded = le_food.transform([food_item])[0]
            category_encoded = le_category.transform([category])[0]
            prediction = model.predict([[food_encoded, category_encoded]])[0]
            prediction_text = f"""
Predicted Nutrition Information
--------------------------------
Food Item : {food_item}
Category  : {category}

Calories       : {prediction[0]:.2f} kcal
Protein        : {prediction[1]:.2f} g
Carbohydrates  : {prediction[2]:.2f} g
Fat            : {prediction[3]:.2f} g
Fiber          : {prediction[4]:.2f} g
Sugars         : {prediction[5]:.2f} g
Sodium         : {prediction[6]:.2f} mg
Cholesterol    : {prediction[7]:.2f} mg
Water Intake   : {prediction[8]:.2f} ml
"""
        except ValueError:
            prediction_text = "Error: Food item or category not found!"

    return render_template("index.html",
                           food_items=food_items,
                           categories=categories,
                           prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
