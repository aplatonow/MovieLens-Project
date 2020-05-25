###############################################################################################
# Project: MovieLens
###############################################################################################
# by Andrii Platonov 




################################
# Create edx set, validation set
################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")




##################################################

# Download link to the edx and validation sets for the project to save as local copy: 
# https://drive.google.com/drive/folders/1IZcBBX0OmL9wu9AdzMBFUG8GoPbGQ38D?usp=sharing


# Use readRDS function to generate edx set from local copy

edx = readRDS("/Users/Admin/Documents/edx.rds")
head(edx)

# Use readRDS function to generate validation set from local copy

validation = readRDS("/Users/Admin/Documents/validation.rds")
head(validation)





###########################################
# Analysis of data
###########################################


# Some extra packages we will need for analysis
library(lubridate)
library(ggplot2)
library(dplyr)
library(colorspace)



# 1) Movie ratings

#Number of ratings by Movies
edx %>% count(movieId) %>% mutate(Number_of_ratings = n) %>%
  ggplot(aes(Number_of_ratings)) + 
  geom_histogram( bins=40, color = "black") +
  scale_x_log10() + 
  ggtitle("Number of ratings by Movies") 

# Top-10 movies by rating
edx %>% group_by(title) %>% 
  summarize(mean_rating =  round(mean(rating), digits = 2), Number_of_ratings = n()) %>% 
  top_n(10) %>%  
  arrange(desc(Number_of_ratings))


# Key findings:
# The analysis of movie’s distribution shows that each of movies has very different number of ratings.
# Along with this, some movies have a way more number of ratings than the other.
# This effect can be quite significant and therefore can be used to build the algorithm 
# to predict movie ratings.


# 2) User preferences

#Number of ratings by Users
edx %>% count(userId) %>% mutate(Number_of_ratings = n) %>%    
  ggplot(aes(Number_of_ratings)) + 
  geom_histogram( bins=50, color = "black") +
  scale_x_log10() + 
  ggtitle("Number of ratings by Users") 

# Key findings:
# User’s distribution plot supports the idea that the activity of users is not the same
# and some of users are extremely active in terms of ratings.  
# This effect should be definitely strong to be considered for building the predict algorithm.



# 3) Rating's value

# Distribution of Rating's value
edx %>%
  group_by(rating) %>% 
  summarize(Number_of_ratings = n()) %>%
  ggplot(aes(x = rating, y = Number_of_ratings)) +
  geom_line() + 
  labs(y = "Number of ratings", x = "Rating's value") +
  ggtitle("Distribution of Rating's value")


#Percentage of Ratings by Rating value
edx %>%
  group_by(rating) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(sumrate = sum(n), percentage = n/sumrate) %>%
  arrange(-percentage) %>%
  ggplot(aes(reorder(rating, percentage), percentage, fill= percentage)) +
  geom_bar(stat = "identity") + coord_flip() +
  labs(y = "Percentage", x = "Rating value") +
  ggtitle("Percentage of Ratings by Rating value")

# Key findings:
# The plot shows that the ratings distribution in not equal across the value of rating. 
# The most popular ratings are 4.0 and 3.0 which are above 50% of all ratings.



# 4) Timestamp analysis

# Formating column "timestamp" for further analysis
edx_timeline <- mutate(edx, year = year(as_datetime(timestamp)))
head(edx_timeline)

# The average rating is experienced a declining trend.
edx_timeline %>% group_by(year) %>%
  summarize(avarage_rating = mean(rating)) %>%
  ggplot(aes(year, avarage_rating)) +
  geom_point() +
  geom_smooth() +
  labs(x = "Years", y = "Avg. rating") +
  ggtitle("Ratings by years")

# Distribution of Ratings value by years
edx_timeline %>% group_by(year, rating) %>%
  summarize(n()) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()+
  labs(x = "Years", y = "Rating's value") +
  ggtitle("Ratings value by years")


# Key findings:
# Analysis of time affect on ratings shows that lower avarage ratings were driven by 
# more ratings with half star ratings.








##############################################
# Identification of optimal methods and tools
##############################################


# Data preparation: generating train and test sets from edx set

y <- edx$rating

set.seed(755, sample.kind="Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
test_set <- edx[test_index, ]
train_set <- edx[-test_index, ]

# Make sure userId and movieId in train set are also in test set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

head(train_set)

head(test_set)



###################################################
# RMSE fuction code 

# This RMSE fuction will be used to evaluate predictions of models
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



##################################################################################
# 1. Average rating Model

# Idemtidying avarage effect
mu <- mean(train_set$rating) 
mu

# Evaluating the RMSE of the model on test set
model_1_rmse <- RMSE(test_set$rating, mu) 
model_1_rmse

# RMSE results of Average rating model 
rmse_results <- tibble(method = "Average rating Model", RMSE = model_1_rmse)
rmse_results      



##################################################################################
# 2. Movie Effect Model

# Idemtidying Movie effect
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))  
  
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 15, data = ., color = I("black"))

# Predicting rating results on test set
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# Evaluating the RMSE of the model on test set
model_2 <- RMSE(test_set$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                     RMSE = model_2 ))

# RMSE results of Movie Effect model
rmse_results %>% knitr::kable()
rmse_results



#################################################################################
# 3. Movie + User Effects Model


# Idemtidying Movie effect 
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Idemtidying User effect (based on the average rating for user with more than 100 movies ratings)
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "blue")

# Culculating user effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predicting rating results on test set
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Evaluating the RMSE of the model on test set
model_3 <- RMSE(test_set$rating, predicted_ratings)
model_3

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                     RMSE = model_3 ))


# RMSE results of Movie + User Effects Model
rmse_results %>% knitr::kable()
rmse_results



############################################################
# 4. Movie + User + Genre Effects Model

# Modify the genres colomn in train and test sets (column splitting)
train_set_g <- train_set  %>% separate_rows(genres, sep = "\\|")

test_set_g <- test_set  %>% separate_rows(genres, sep = "\\|")


# Idemtidying Movie effect 
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Idemtidying User effect 
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Idemtidying Genre effect 
genres_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Predicting rating results on test set
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(preddd = mu + b_i + b_u + b_g) %>%
  .$preddd

# Evaluating the RMSE of the model on test set
model_4 <- RMSE(test_set$rating, predicted_ratings)
model_4

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre Effects Model",  
                                     RMSE = model_4 ))
rmse_results



#################################################################
# 5. Regularized Movie + User + Genre Effects Model


#####
# 5a) Indentification of lambda as a tuning parameter for Regularized Model

lambdas <- seq(3, 6, 0.25)

# Cross-validation will be used for the best lambda indentification 

rmses <- sapply(lambdas, function(l){
  
  # idemtidying avarage effect on training set
  mu <- mean(train_set$rating)
  
  # idemtidying movie effect on training set
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # idemtidying user effect on training set
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Idemtidying Genre effect on training set
  b_g <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u))
  
  # Predicting rating results on test set
  predicted_ratings <- test_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(preddd = mu + b_i + b_u + b_g) %>%
    .$preddd
  
  return(RMSE(test_set$rating, predicted_ratings))
})
# Plot rmses vs lambdas to identify the optimal lambda
qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda   # the best lambda to be used for the regularized model




#######
# 5b) Compute regularized model using the best lambda (= 4.75)

lambda_2 <- 4.75

# idemtidying avarage effect on training set
mu <- mean(train_set$rating)
# Compute regularized estimates of b_i using lambda on training set
movie_reg <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda_2), n_i = n())
# Compute regularized estimates of b_u using lambda on training set
user_reg <- train_set %>% 
  left_join(movie_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_2), n_u = n())
# Compute regularized estimates of b_g using lambda on training set
genre_reg <- train_set %>%
  left_join(movie_reg, by='movieId') %>%
  left_join(user_reg, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda_2), n_g = n())
# Predict ratings on test set
predicted_ratings <- test_set %>% 
  left_join(movie_reg, by='movieId') %>%
  left_join(user_reg, by='userId') %>%
  left_join(genre_reg, by = 'genres') %>%
  mutate(pred_reg = mu + b_i + b_u + b_g) %>% 
  .$pred_reg
# Evaluating the RMSE of Regularized model's algorithm
model_5r <- RMSE(test_set$rating,predicted_ratings)
model_5r

# RMSE results of Regularized model
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre Effects Model",  
                                     RMSE = model_5r))

rmse_results

rmse_results %>% knitr::kable()
rmse_results


#########################
# 6) Regularized Movie + User + Genre Effects Model on validaion set

# Final Algorithm evaluation using the best lambda

lambda_2 <- 4.75

# idemtidying avarage effect on edx set
mu <- mean(edx$rating)
# Compute regularized estimates of b_i using lambda on edx set
movie_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda_2), n_i = n())
# Compute regularized estimates of b_u using lambda on edx set
user_reg <- edx %>% 
  left_join(movie_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_2), n_u = n())
# Compute regularized estimates of b_g using lambda on edx set
genre_reg <- edx %>%
  left_join(movie_reg, by='movieId') %>%
  left_join(user_reg, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda_2), n_g = n())
# Predict ratings on validation set
predicted_ratings <- validation %>% 
  left_join(movie_reg, by='movieId') %>%
  left_join(user_reg, by='userId') %>%
  left_join(genre_reg, by = 'genres') %>%
  mutate(pred_reg = mu + b_i + b_u + b_g) %>% 
  .$pred_reg
# Evaluating the RMSE of final algorithm
model_6_VAL <- RMSE(validation$rating,predicted_ratings)
model_6_VAL


# RMSE results of final algorithm
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre Effects Model VAL",  
                                     RMSE = model_6_VAL ))



###################################################
# Final results of RMSE 
###################################################

rmse_results %>% knitr::kable()
rmse_results

