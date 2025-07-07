import streamlit as st

@st.cache_data
def load_model():
    # 🔹 Load the CSV file
    df = pd.read_csv("News_Category_Dataset.csv")  # Make sure file name matches exactly

    # 🔹 Check expected columns
    if 'headline' not in df.columns or 'category' not in df.columns:
        st.error("Dataset must have 'headline' and 'category' columns.")
        st.stop()

    # 🔹 Keep only selected categories
    top_categories = ['POLITICS', 'ENTERTAINMENT', 'BUSINESS', 'SPORTS', 'TECH']
    df = df[df['category'].isin(top_categories)]

    # 🔹 Clean the text
    df['cleaned'] = df['headline'].apply(clean_text)

    # 🔹 Encode category labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # 🔹 Split into training/testing
    X = df['cleaned']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 🔹 Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, le
