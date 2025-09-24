# 🛒 E-Commerce Transactions Analysis | تحلیل تراکنش‌های فروشگاه اینترنتی

## 📖 Project Description | توضیحات پروژه
**English:**  
This project analyzes an *E-Commerce Transactions Dataset* (50,000 records) with exploratory data analysis, preprocessing, and a baseline model using Logistic Regression. While the model showed weak performance due to low correlation between features and target, the work highlights the essential role of **feature engineering** in predictive modeling.  

**فارسی:**  
این پروژه به تحلیل یک *دیتاست تراکنش‌های فروشگاه اینترنتی* (۵۰ هزار رکورد) می‌پردازد. مراحل شامل تحلیل اکتشافی، پیش‌پردازش و مدل پایه با رگرسیون لجستیک است. با وجود عملکرد ضعیف مدل به دلیل همبستگی پایین ویژگی‌ها و هدف، پروژه اهمیت **فیچر انجینیرینگ** را در مدل‌سازی پیش‌بینانه نشان می‌دهد.  

---

## 📊 Data Overview | مرور داده‌ها
**English:**  
- Records: 50,000  
- Columns: 8 → Transaction ID, User Name, Age, Country, Product Category, Purchase Amount, Payment Method, Transaction Date  
- Data is complete, no missing values.  
- Key variables:  
  - Age (18–70)  
  - Purchase Amount (5–1000)  
  - Balanced product categories (~6.2k each)  

**فارسی:**  
- تعداد رکوردها: ۵۰ هزار  
- ستون‌ها: ۸ → شناسه تراکنش، نام کاربر، سن، کشور، دسته محصول، مبلغ خرید، روش پرداخت، تاریخ تراکنش  
- داده‌ها کامل هستند و مقدار خالی ندارند.  
- متغیرهای کلیدی:  
  - سن (۱۸ تا ۷۰)  
  - مبلغ خرید (۵ تا ۱۰۰۰)  
  - دسته‌های محصول متعادل (هرکدام حدود ۶.۲ هزار رکورد)  

---

## 🔍 Exploratory Data Analysis (EDA) | تحلیل اکتشافی
**English:**  
- Sales are balanced across countries and product categories.  
- Yearly sales peak in 2024, with fewer records in 2025 (incomplete year).  
- Age distribution is fairly uniform.  
- Payment methods are used in almost equal proportion.  
- No strong relationship found between age, amount, and product choice.  

**فارسی:**  
- فروش بین کشورها و دسته‌های محصول متعادل است.  
- بیشترین فروش در سال ۲۰۲۴ ثبت شده؛ سال ۲۰۲۵ رکورد کمتری دارد (احتمالاً ناقص).  
- توزیع سنی کاربران نسبتاً یکنواخت است.  
- روش‌های پرداخت تقریباً به طور برابر استفاده شده‌اند.  
- رابطه قوی بین سن، مبلغ خرید و انتخاب دسته محصول دیده نشد.  

---

## ⚙️ Preprocessing | پیش‌پردازش
**English:**  
- Converted transaction date into `Year`, `Month`, `Day`.  
- Dropped unnecessary columns (Transaction ID, User Name, raw date).  
- One-hot encoded `Country` and `Payment Method`.  
- Label-encoded `Product Category` → `Product_Category_Label` as target.  

**فارسی:**  
- تبدیل تاریخ تراکنش به ستون‌های `Year`، `Month`، `Day`.  
- حذف ستون‌های غیرضروری (شناسه تراکنش، نام کاربر، تاریخ خام).  
- وان‌هات برای ستون‌های `Country` و `Payment Method`.  
- برچسب‌گذاری دسته محصول (`Product_Category_Label`) به عنوان متغیر هدف.  

---

## 🤖 Model Training | آموزش مدل
**English:**  
- Model: Multinomial Logistic Regression  
- Accuracy: ~0.125 (close to random guessing across 8 classes)  
- Precision/Recall/F1 scores were low, confirming weak predictive power.  

**فارسی:**  
- مدل: رگرسیون لجستیک چندکلاسه  
- دقت: حدود ۰.۱۲۵ (نزدیک به حدس تصادفی بین ۸ کلاس)  
- مقادیر Precision/Recall/F1 پایین بود و نشان داد مدل قدرت پیش‌بینی کافی ندارد.  

---

## 📉 Correlation & Conclusion | همبستگی و جمع‌بندی
**English:**  
- Correlation heatmap shows very weak feature–target relationships.  
- Current dataset provides limited predictive signals.  
- **Key takeaway:** Feature engineering is more important than model choice. Without meaningful features, even advanced models cannot achieve strong results.  

**فارسی:**  
- Heatmap همبستگی نشان داد که رابطه ویژگی‌ها با هدف بسیار ضعیف است.  
- دیتاست فعلی سیگنال کافی برای پیش‌بینی دقیق فراهم نمی‌کند.  
- **نکته کلیدی:** فیچر انجینیرینگ از انتخاب مدل مهم‌تر است. بدون ویژگی‌های معنادار، حتی پیشرفته‌ترین مدل‌ها هم نتیجه مطلوب نخواهند داشت.  

---
