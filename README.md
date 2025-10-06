# Machine Learning for Unit Commitment - Assignment 2
This project applies machine learning to predict power plant commitment decisions in day-ahead generation scheduling.  
By training classification models on optimized/historical unit commitment results, it explores how data-driven approaches can complement traditional optimization for faster scheduling.

**Date:** October-November 2024

---

## 👥 Contributors
[@strenchev](https://github.com/strenchev)  
[@nic0lew0ng](https://github.com/nic0lew0ng)  
[@raullabarthes](https://github.com/raullabarthes)  

---

## 📌 Project Overview
The study investigates how **linear and non-linear classifiers** (SVMs) can approximate the binary on/off status of generation units from a system operator perspective.  
Data include load, wind, and weather features, used to train and evaluate models predicting optimal commitment states.  
The workflow covers:
- Data preprocessing and feature generation  
- Model training and comparison across kernel types  
- Evaluation and interpretation of prediction accuracy  

---

## 🛠 Skills & Techniques Demonstrated
- Unit commitment modeling and operational planning  
- Linear and non-linear SVM classification  
- Feature engineering with meteorological and load data  
- Model evaluation using precision, recall, and F1-score  
- Comparative analysis of kernel performance and regularization  

---

## 📈 Key Results
- Classification models can accurately replicate unit commitment results using only historical data.  
- Non-linear kernels (e.g., RBF) generally outperform linear ones for capturing operational boundaries.  
- Weather and demand features significantly improve model performance.  
- Classification can accelerate scheduling when integrated into optimization workflows.  

---

## 🚀 Optional Improvements
- Address class imbalance to better predict rare commitment states.  
- Explore hybrid ML–optimization schemes for real-time applications.  
- Extend feature space to include market prices and reserve margins.  

---

## 🧠 Key Files
- **[`A2.py`](A2.py):** Implements the mathematical formulation and optimization model for unit commitment, including startup, ramping, and min up/down time constraints.  
- **[`assignment2_datapreprocessing.ipynb`](assignment2_datapreprocessing.ipynb):** Data cleaning, feature engineering, and exploratory analysis.  
- **[`assignment2_step3.ipynb`](assignment2_step3.ipynb):** Training and evaluating linear vs. non-linear classifiers, with visual comparisons.

---

## 📝 References
- Machine Learning for Energy Systems course materials.
- Assignment instructions are provided in [Assignment 2.pdf](Assignment%202.pdf).  

---

### 🔖 Tags
`#machinelearning` `#optimization` `#svm` `#classification` `#unitcommitment` `#energysystems`


