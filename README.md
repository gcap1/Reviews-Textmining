Customer Review Sentiment Analysis
Overview
This project analyzes customer reviews from e-commerce dataset, consisting of around 10,000 entries. Each review includes:

clothes ID
customer age
content of the review
polarity (1 if the customer recommends the product, 0 otherwise)
The primary goal is to predict whether a customer would recommend a product based on their review text.

Steps
Step 1: Import Data and Build Corpus
Load the data and create a corpus for text mining.

# Load data
reviews <- read.csv("reviews.csv")

# Required Libraries
library(tm)
Step 2: Clean and Pre-process the Text Data
Prepare the data for text mining by applying the following transformations:

Convert to lowercase
Remove numbers
Remove punctuation
Remove stopwords
Remove extra whitespaces
Apply stemming
# Text cleaning process
reviews_corpus <- Corpus(VectorSource(reviews$content))
reviews_corpus <- tm_map(reviews_corpus, content_transformer(tolower))
reviews_corpus <- tm_map(reviews_corpus, removeNumbers)
reviews_corpus <- tm_map(reviews_corpus, removePunctuation)
reviews_corpus <- tm_map(reviews_corpus, removeWords, stopwords("en"))
reviews_corpus <- tm_map(reviews_corpus, stripWhitespace)
reviews_corpus <- tm_map(reviews_corpus, stemDocument)
Step 3: Generate the TF-IDF Matrix
Create the Term Frequency-Inverse Document Frequency (TF-IDF) matrix.

# Create the TF-IDF matrix
dtm <- DocumentTermMatrix(reviews_corpus, control = list(weighting = weightTfIdf))
Step 4: Reduce Term Dimensions by Removing Sparse Terms
To reduce the matrix size, filter terms with a sparsity threshold greater than 0.93.

# Remove sparse terms
tfidf <- removeSparseTerms(dtm, 0.93)
inspect(tfidf)
Step 5: Predictive Modeling – Classification
Create a predictive model to classify whether a review recommends the product.

Data Preparation
# Prepare data for modeling
review_df <- data.frame(as.matrix(tfidf), Recommended = reviews$Recommended)
Split Data for Training and Validation
Split the dataset into 60% training and 40% validation sets with a fixed random seed for reproducibility.

set.seed(1)
train_index <- sample(seq_len(nrow(review_df)), size = 0.6 * nrow(review_df))
train_data <- review_df[train_index, ]
validation_data <- review_df[-train_index, ]
Fit Logistic Regression Model
Use logistic regression (GLM) to fit the model.

# Train logistic regression model
model <- glm(Recommended ~ ., data = train_data, family = "binomial")
Predict and Evaluate Model
Predict on the validation set and generate a confusion matrix.

# Predict probabilities
predicted_prob <- predict(model, newdata = validation_data, type = "response")

# Set cutoff for prediction
predicted_class <- ifelse(predicted_prob > 0.5, 1, 0)

# Confusion matrix
table(Predicted = predicted_class, Actual = validation_data$Recommended)
Step 6: Generate a Word Cloud (Optional)
Visualize the most important words in the reviews using a word cloud.

# Libraries required
library(wordcloud)
library(RColorBrewer)

# Convert tfidf to matrix and sum term frequencies
m <- as.matrix(tfidf)
v <- sort(colSums(m), decreasing = TRUE)

# Generate word cloud
wordcloud(names(v), v, random.order = FALSE, max.words = 100, colors = brewer.pal(8, "Dark2"))
Wordcloud Options:
min.freq: Minimum frequency for words to appear in the cloud.
max.words: Maximum number of words to display.
random.order: Display words in random order (default TRUE).
rot.per: Proportion of words displayed vertically.
colors: Color scale for word frequency.

Example Word Cloud
Here’s an example of the generated word cloud from the dataset:
![download](https://github.com/user-attachments/assets/41703180-a7c1-438e-a2a4-1042c20b7660)
