# app.py
# Amirali – Titanic Survival Predictor with Streamlit
# هدف: ساخت یک داشبورد تمیز و شبیه نمونه‌های حرفه‌ای برای رزومه

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# تنظیمات کلی صفحه استریم‌لیت
# -----------------------------
# توضیح برای خودم: این فانکشن ظاهر کلی صفحه (عنوان تب، آیکن، layout) رو تعیین می‌کنه
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------
# عنوان اصلی و توضیح بالا
# -----------------------------
# توضیح برای خودم: بخش هدر اصلی اپلیکیشن؛ همیشه بالای صفحه دیده می‌شه
st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 0;">🚢 Titanic Survival Predictor</h1>
    <p style="text-align: center; font-size: 16px; color: #cccccc;">
        Enter passenger details to predict survival probability on the Titanic.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# بارگذاری مدل آموزش‌دیده
# -----------------------------
# توضیح برای خودم: این قسمت مدل ذخیره شده در پوشه models رو می‌خونه
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voting_clf_model.pkl")

@st.cache_resource
def load_model():
    # توضیح برای خودم: اگر فایل مدل وجود نداشت، ارور کاربرپسند می‌دم
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Please train the model and save it to the 'models/voting_clf_model.pkl' path.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# اگر مدل لود نشده، ادامه نده
if model is None:
    st.stop()

# -----------------------------
# سایدبار – ورودی‌های مسافر
# -----------------------------
# توضیح برای خودم: این بخش ظاهر سایدبار سمت چپ رو می‌سازه
with st.sidebar:
    st.markdown("## Passenger Information")
    st.write("Fill in the passenger details below:")

    # کلاس بلیت
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)

    # جنسیت
    sex = st.selectbox("Sex", ["male", "female"])

    # سن
    age = st.slider("Age", min_value=0, max_value=80, value=24)

    # تعداد هم‌سفر/همسر
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)

    # تعداد والدین/فرزندان
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)

    # هزینه بلیت
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=512.0, value=34.5, step=0.5)

    # بندر سوار شدن
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

# -----------------------------
# ساخت دیتافریم اولیه از ورودی‌ها
# -----------------------------
# توضیح برای خودم: این DataFrame دقیقا ورودی خام کاربره
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
}])

# -----------------------------
# Feature Engineering – انطباق با مدل
# -----------------------------
# توضیح برای خودم: این قسمت باید دقیقا همون فیچرهایی رو بسازه که تو نوت‌بوک training.ipynb ساختم

# FamilySize = تعداد اعضای خانواده + خود شخص
input_data["FamilySize"] = input_data["SibSp"] + input_data["Parch"] + 1

# IsAlone = اگر تنها سفر کرده باشد (FamilySize = 1)
input_data["IsAlone"] = (input_data["FamilySize"] == 1).astype(int)

# FarePerPerson = تقسیم هزینه بلیت بر تعداد اعضای خانواده
input_data["FarePerPerson"] = input_data["Fare"] / input_data["FamilySize"]

# Title = چون اسم نداریم، بر اساس جنسیت یک عنوان منطقی می‌گذارم
input_data["Title"] = "Mr"
input_data.loc[input_data["Sex"] == "female", "Title"] = "Mrs"

# Deck = چون Cabin نیست، یک دسته ثابت "Unknown" می‌گذارم
input_data["Deck"] = "Unknown"

# -----------------------------
# Prediction Section – داشبورد اصلی
# -----------------------------
# توضیح برای خودم: این بخش وسط صفحه رو به سه ستون تقسیم می‌کنه:
# ۱. گیج احتمال بقا
# ۲. چارت اهمیت ویژگی‌ها
# ۳. چارت توزیع احتمال (نمایشی)
st.markdown("---")
st.markdown("### Prediction Dashboard")

try:
    # پیش‌بینی احتمال بقا
    proba = model.predict_proba(input_data)[0][1]   # احتمال کلاس 1 (Survived)
    pred = model.predict(input_data)[0]             # کلاس پیش‌بینی‌شده

    # سه ستون برای گیج و نمودارها
    col_gauge, col_feat_imp, col_prob_dist = st.columns([2, 2, 2])

    # -------------------------
    # ستون اول: گیج احتمال بقا
    # -------------------------
    with col_gauge:
        st.markdown("#### Survival Probability")

        # توضیح برای خودم: برای حس گیج، از یک نوار پیشرفت افقی و عدد درصد استفاده می‌کنم
        st.markdown(
            f"""
            <div style="font-size: 40px; font-weight: bold; margin-bottom: 10px;">
                {proba * 100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # نوار پیشرفت برای حس بصری گیج
        st.progress(float(proba))

        # متن توضیحی
        if proba >= 0.7:
            st.success("High chance of survival.")
        elif proba >= 0.4:
            st.info("Moderate chance of survival.")
        else:
            st.warning("Low chance of survival.")

        # نتیجه نهایی به صورت بنر سبز/قرمز
        if pred == 1:
            st.success("Result: Passenger is predicted to SURVIVE 🎉")
        else:
            st.error("Result: Passenger is predicted NOT TO SURVIVE 😔")

    # --------------------------------
    # ستون دوم: Feature Importance mock
    # --------------------------------
    with col_feat_imp:
        st.markdown("#### Feature Importance (demo)")

        # توضیح برای خودم:
        # اگر مدل اصلی‌ام در نوت‌بوک feature_importances_ داشت، می‌تونم اینجا واقعی‌شو نمایش بدم.
        # فعلاً برای داشبورد رزومه، یک بارچارت با اعداد نمایشی می‌سازم.

        feat_names = ["Sex (female)", "Pclass", "Age", "Fare"]
        feat_values = [0.40, 0.30, 0.20, 0.10]  # این‌ها فقط برای نمایش هستند

        feat_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": feat_values
        })

        st.bar_chart(feat_df.set_index("Feature"))

    # ---------------------------------------
    # ستون سوم: Probability Distribution mock
    # ---------------------------------------
    with col_prob_dist:
        st.markdown("#### Probability Distribution (demo)")

        # توضیح برای خودم:
        # این قسمت صرفاً برای زیبایی UX است؛ یک توزیع مصنوعی حول proba می‌سازم
        samples = np.random.normal(loc=proba, scale=0.08, size=500)
        samples = np.clip(samples, 0, 1)

        prob_df = pd.DataFrame({"Survival Probability": samples})
        st.area_chart(prob_df)

        st.caption("Synthetic distribution around current prediction (for visualization).")

    # -----------------------------
    # خلاصه پروفایل مسافر پایین صفحه
    # -----------------------------
    st.markdown("---")
    st.markdown("#### Passenger Profile")

    # توضیح برای خودم: این بخش خلاصه ورودی‌ها رو در قالب متن مرتب نشان می‌دهد
    profile_text = (
        f"Age {age}, "
        f"{sex.capitalize()}, "
        f"Class {pclass}, "
        f"Fare ${fare:.2f}, "
        f"FamilySize {int(input_data['FamilySize'].iloc[0])}, "
        f"Port {embarked}"
    )
    st.markdown(profile_text)

except Exception as e:
    # توضیح برای خودم: اگر هر اروری در زمان predict ایجاد شد، اینجا نمایش داده می‌شه
    st.error(f"Error during prediction: {e}")
