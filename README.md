# 🍽️ Restaurant Recommender

A smart and intuitive Restaurant Recommendation System built using Flask and Machine Learning, developed as part of my Internship (Task 2).

This system analyzes cuisines, user preferences, ratings, cost, votes, and city location to provide highly accurate restaurant suggestions.  
It uses a custom Hybrid Weighted Model—similar to modern food delivery apps—to generate realistic and personalized recommendations.

The project also features a clean UI, auto-suggestions, and Google Maps integration for easy restaurant navigation.

---

## 🚀 Features
- TF-IDF Cuisine Matching  
- Hybrid Weighted Scoring (cuisine + rating + votes + cost + city)  
- City-Based Filtering  
- Auto-Suggestions for Cuisines  
- Google Maps “View” Button  
- Modern & Clean UI  

---

## 🛠️ Tech Stack
- Python, Flask  
- Pandas, NumPy  
- Scikit-Learn  
- HTML, CSS, JavaScript  

---

## 📂 Project Structure
Restaurant-Recommender/
│
├── app.py  
├── suggestions.json  
├── Dataset.csv  
├── requirements.txt  
├── Procfile  
├── runtime.txt  
│
├── templates/  
│   ├── index.html  
│   ├── result.html  
│   ├── header.html  
│
└── static/  
    ├── style.css  
    └── app.js

---

## ▶️ Run Locally
pip install -r requirements.txt
python app.py

App will run at:  
**http://127.0.0.1:5000**

---

## 👨‍💻 Author
**Vishal Akale**  
GitHub: https://github.com/VishalAkale
