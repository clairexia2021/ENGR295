# ENGR295 Master Project
#### Spring 2023

### Statement of Work
1. Extract data using web scraping from IMDb, scope is the most 50 relevant reviews for the most recent 100 movies in the United States.
2. Preprocess text 
   1. Stopword filtering: remove the most common words that have no added information to the text
   2. Stemming: produce morphological variants of a root/base word. 
3. Perform word embedding and position embedding
4. Implement transformer and multi-head attention
   1. Utilize BERT architecture and its implementation in the transformers library 
5. Implement deep learning models based output from transformers
   1. Regression layer model including optimizer, scheduler and loss function
   2. Text summarization with BART to understand language including text length control and BartTokenizer to make tokens from the text sequence
6. Model Evaluation in regression model
   1. Mean square error is an absolute measure of the goodness of fit, which means how much the predicted results deviate from the actual numbers. The best regression model is the one with the lowest MSE.
   2. R squared measures how much variability in dependent variables can be explained by the model. It is the square of the Correlation Coefficient. It is between 0 to 1 and a bigger value indicates a better fit between prediction and actual value.
7. Text Data visualization with qualitative comparisons, sentimental analysis, and prediction result comparison. Graphs, charts, word clouds can be used to showcase text data in a visual manner
   1. Distributions of both predicted and actual ratings, text lengths, word counts and the most common n words (before and after removing stop words) can be visualized.
   2. Some graph techniques such as histogram, bubble chart, bar chart, heatmap can be utilized. 
