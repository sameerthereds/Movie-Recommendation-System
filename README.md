# 🎬 **Movie Recommendation System**

## 📌 Overview  
This project implements an **end-to-end movie recommendation system** with an interactive user interface. It combines **content-based filtering** and **collaborative filtering** techniques to provide **personalized movie recommendations**. Users can interact with the system through an intuitive UI to receive movie suggestions based on their preferences.

### 📄 **Key Features**
- ✅ **End-to-End Recommendation System**: Handles data collection, preprocessing, model training, and real-time movie recommendations.  
- ✅ **Interactive UI**: Allows users to input preferences and receive **personalized movie recommendations**.  
- ✅ **Content-Based Filtering**: Recommends movies based on features like **genre, director, and cast**.  
- ✅ **Collaborative Filtering**: Suggests movies based on user ratings and similarity between users.  

---

## 🎯 **Introduction**  
A **movie recommendation system** predicts movies users might like based on their past preferences. There are two primary approaches:  
1️⃣ **Collaborative Filtering**: Finds similar users and recommends movies highly rated by them.  
2️⃣ **Content-Based Filtering**: Identifies features (genres, directors, etc.) that the user prefers and recommends similar movies.  

This project primarily utilizes the **content-based approach** to build a **personalized recommendation engine**.

---

## 📊 **Data Preparation**  
We used the **ml-latest-small dataset** from **MovieLens**, along with additional **IMDB metadata**.  
- **Datasets Used**:
  - 📂 `ratings.csv`, `movies.csv`, `links.csv`, `genome-tags.csv`, `genome-scores.csv` (MovieLens)
  - 📂 `name.basics.tsv`, `title.crew.tsv` (IMDB)  
- **Processing Steps**:
  - 🏗️ Data loaded into **Pandas DataFrames**.
  - 🔄 **Preprocessing**: Data cleaned and formatted for **feature extraction**.
  - 📉 **Dimensionality Reduction**: Applied **PCA (Principal Component Analysis)** to reduce feature size while retaining 95% variance.

---

## 🛠️ **Solution Approach**  
### **1️⃣ User-Behavior Matrix**  
A **user-movie rating matrix** was created from `ratings.csv` to represent user preferences. If a movie wasn’t rated, it was assigned **0**.

### **2️⃣ Movie-Feature Matrix**  
- Extracted **1149 features** (1128 tags + 20 genres + release year).  
- Applied **PCA**, reducing features to **585 principal components**.  

### **3️⃣ User-Movie Matrix (Final Predictions)**  
- Computed **User-Feature Matrix** by multiplying **User-Behavior Matrix** × **Movie-Feature Matrix**.  
- **Final predictions** made by sorting movies based on user preference scores.  

---

## 🎥 **Recommendation Strategies**  
### **🔹 For Existing Users**
- Predicts movies based on their previous ratings using **user-movie preference scores**.
- Recommends **top 25 movies** the user hasn’t rated yet.

### **🔹 For New Users**
- Users first **select five favorite genres**.  
- They rate highly-rated movies from these genres.  
- System **infers their taste** and recommends movies **similar to their choices**.

### **🔹 Additional Recommendations**
- Uses **IMDB metadata** to recommend movies from **the same director** as the top recommended movies.

---

## 🎨 **User Interface**
The **front-end** is built using the **Dash framework** with a **Single Page Application (SPA) design**.

### 📌 **UI Layouts**  
1️⃣ **User Login / Sign-Up**:  
   - Existing users enter **User ID** to log in.  
   - New users click **“New User”** to create an account.  

2️⃣ **Genre Selection** (For New Users):  
   - Users select **five favorite genres** before proceeding.  

3️⃣ **Movie Recommendation & Rating**:  
   - Displays **25 recommended movies**.  
   - Users can **rate movies to refine future recommendations**.  
   - **Additional recommendations** are based on **movie directors**.  
   - Users can refresh recommendations **after rating movies**.  

---

## 🔬 **Evaluation & Results**
- **Verified** the system by comparing features of recommended movies with those of highly rated movies.  
- Checked results for both **new and existing users**.  
- Observed **high relevance** in recommendations, confirming the effectiveness of **content-based filtering**.  

---

## 🏁 **Conclusion & Future Improvements**
- The **PCA-based approach** enhanced efficiency by reducing dimensions while maintaining accuracy.  
- The recommendation system successfully **tailors movie suggestions to user preferences**.  
- **Potential improvements**:
  - 🏗️ Improve data **preprocessing** for larger datasets.  
  - ⚡ Optimize **execution speed**.  
  - 🤝 **Hybrid approach**: Combine **collaborative filtering** with **content-based filtering** for better recommendations.  

---

## 🛠 **Tech Stack**
- **Backend**: Python (Pandas, NumPy, Scikit-learn)  
- **Frontend**: Dash (Plotly), HTML/CSS  
- **Data Processing**: Pandas, NumPy  
- **Machine Learning**: PCA, Cosine Similarity  
- **Database**: MovieLens, IMDB  

---

## 🚀 **Getting Started**
To run the project locally:  

```bash
# Clone the repository
git clone https://github.com/sameerthereds/Movie-Recommendation-System.git

# Navigate to the project directory
cd Movie-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

