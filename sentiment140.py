import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from collections import Counter
import re

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv(r'H:\NLPCourse\Class Practices\sentiment140.csv', encoding='latin-1', header=None, usecols=[0, 5], names=['sentiment', 'text'])

# Handle missing data
df.dropna(subset=['text'], inplace=True)

# Print total tweets and average lengths
print("Total Tweets are:", len(df))
print("Average length is:", df["text"].apply(len).mean())
print("Median length is:", df["text"].apply(len).median())

# Display stopwords
stop_words = set(stopwords.words("english"))
print("Stopwords:", stop_words)

# Set of punctuation
punctuation = set(string.punctuation)
print("Punctuation:", punctuation)

# Text cleaning function
def clean_text_with_re(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

# Clean the text
df['clean_text'] = df['text'].apply(clean_text_with_re)

# Find most common words
words = ' '.join(df['clean_text']).split()
common_word = Counter(words).most_common(10)
print("Most common words:", common_word)

# Visualize sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['blue', 'skyblue'])
plt.title("Sentiment Distribution")
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# WordCloud with neon blue and black theme
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="Blues").generate(' '.join(words))

# Display WordCloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
