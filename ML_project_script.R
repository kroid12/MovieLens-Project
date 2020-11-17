##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#### 1. Comprehensive analysis of the database ####
# I omit the part we create edx set and validation set (final hold-out test set). I did it in a separate script document for not having it here.
# so let´s move straight forward to review what´s in the edx set:

str(edx)

#it shows 900005 obs and 6 variables. the first thing I notice is that R reads some variables as numeric (userId, movieId) when they´re don´t, and also the timestamp variable can also be arrange to date; 
#so we need to change that first:

edx$userId <- as.character(edx$userId)
edx$movieId <- as.character(edx$movieId)
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
edx$timestamp <- as_datetime(edx$timestamp, origin = "1970-01-01")

# now, we can make some arranngements here with some variables which cwe can break down in some other variables, 
# for instance, year is in the title and the are several genres joined in most of the observations. so let´s make some feature engineering:
# we will start with the genres column. let´s see how it is 
tibble(count = str_count(edx$genres, pattern = "|"), genres = edx$genres) %>% 
  group_by(count, genres) %>%
  summarise(n = n()) %>%
  arrange(-count) %>% 
  head()
# It shows that a great deal of movies are classiffied with several genres, and 256 up to 8 differrent genres, 
# these clearly could be split out. We can keep the first genre as it is the most important one and not all movies have been cathegorized with two or more genres
edx <- edx %>% separate(genres, c("genre", "genre2"), sep = "[|]", fill = "warn") %>% select(-genre2)
edx$genre <- as.factor(edx$genre)
#of course not all new genres need to be used but to have them separated could help, so, we can consider the first genre as the main and use only that column.
# we can take a quick view and see some tendences.
edx %>% group_by(genre) %>%
  summarize(count = n()) %>%
  top_n(10) %>%
  arrange(desc(count)) 
# Action appears to be the most rated genre, folowed by Comedy and Drama. 
# Moving on, we arrange the title column and separate the year into a different column:
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# Now we can take a closer look to see how the year performs within the data

edx %>% group_by(year) %>% 
  summarize(count =n()) %>% 
  arrange(desc(count)) %>%
  select(year, count)

# We can also see there are years with most review of others, starting with 1995. 
# This tells us that clearly there is a trend for rating and this could depend on other variables, such blockbusters and when they were launched.
# we can see this association in the following graph, taking as example the top 5 genres rated:

genre_year <- edx %>%
  select(movieId, genre, year) %>% 
  group_by(year, genre) %>% 
  summarise(number = n())

genre_year %>%
  filter(year > 1970) %>%
  filter(genre %in% c("Action", "Comedy", "Drama", "Adventure", "Crime")) %>%
  ggplot(aes(x = year, y = number)) +
  geom_line(aes(color=genre))

#It shows as expected, the most rated genres are concentrated mostly in the 90`s, 
# and we can explore how this trend is correlated with the ratings given by the users:

edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_line() 

# Interestly, we see that users who have more classic-taste on movies are less and trend to rate higher those movies, in contrast with a broader set of modern-taste users, who trend to give lower average ratings. 
# This is more clear to see in the folowing graph: 
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# As it shows, the distribution is skewed and could affect our prediction; so we need to consider that all user are different (user effect) 
# and some of them give few ratings, but in general, the ratings distribution is close to normal:

edx %>% group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black") + 
  ggtitle("User Effect Distribution") +
  xlab("User Bias") +
  ylab("Count")

# So in general, we see that there is 4 major effects in the data set: user, genre year and of course, the movie. 
# All this features will be considered to build our model and explained in the model section.

#### 2.2 Model ####

# As it was described in the Machine Learning course, let´s create the Root Mean Squared Error (RMSE) function: 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# based in the analysis made before, we will use a linear model based on the formula

# $y =  \mu + b_{i} + b_{u} + b_{g} + b_{y} + \epsilon_{i,u,g,y}$
  
# where starting form the simplest model: the mean $$\mu$$ of the ratings, which considers that all users give the same rating to every movie. 
# This of course is not optimal, but according the statistical theory presented in the course, the average minimizes the RMSE, so any model should be based on that. 
# Creating test and training sets from edx

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "genre") %>%
  semi_join(train_set, by = "year")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# #build the simplest model: averaging the ratings
mu <- mean(train_set$rating)
mu
RMSE(test_set$rating, mu)
# So 1.060054 will be our starting point, and we will need to move down to a number bellow 0.8649 to reach our goal.

##### 2.2.1 Movie effect (bi) ####

# As mentioned before, we found that several features could be considered for prediction, and the movie effect is one of them. 
# This can be explained considering that different movies have different ratings distribution as, of course, some are blockbusters 
# and  there  are different user preferences. This movie "bias" or "movie effect" is expressed as $$b_{i}$$ in the following formula (the mean of the difference between the observed rating y and the mean μ):

bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# And we can see it in the following graph

bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count")

# Now let´s see how can we predict the rating with mean + the movie effect $$b_{i}$$  

y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i
RMSE(test_set$rating, y_hat_bi)

# 0.9429615 is a huge jump from our starting model with the average.

##### 2.2.2 User effect (bu) ####

# In the same way we did with the movie effect, we can now build a the user or user effect:

bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict the rating with mean + bi + bu
y_hat_bi_bu <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
RMSE(test_set$rating, y_hat_bi_bu)

# 0.8646843 is beyond the cut of 0.8649 requested as goal for the excersise, however, we can move on to see how far we can get with the other variables considered in our model.

# Gender Effect (bg)
bg <- train_set %>% 
  left_join(bi, by ='movieId') %>%
  left_join(bu, by='userId') %>%
  group_by(genre) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))


# Predict the rating with mean + bi + bu + bg

y_hat_bi_bu_bg <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  left_join(bg, by= "genre") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred
RMSE(test_set$rating, y_hat_bi_bu_bg)

# 0.8645672 is not a big change, but it keeps helping us.
# And finally the Year Effect (by)

by <- train_set %>% 
  left_join(bi, by ='movieId') %>%
  left_join(bu, by='userId') %>%
  left_join(bg, by= "genre") %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))


# Final prediction with rating with all the effects: mean + bi + bu + bg + by

y_hat_bi_bu_bg_by <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  left_join(bg, by= "genre") %>%
  left_join(by, by= "year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  .$pred

RMSE(test_set$rating, y_hat_bi_bu_bg_by)

#### 2.3 Regularization ####
# Considering that all this effects are biased, we need now to regularize the model itself in order to avoid this bias to affect the prediction by penalizing extreme values:
regularization <- function(lambda, trainset, testset){

# Mean
mu <- mean(trainset$rating)
  # Movie effect (bi)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  # User effect (bu)  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
  # Gender Effect (bg)
  b_g <- trainset %>% 
    left_join(bi, by ='movieId') %>%
    left_join(bu, by='userId') %>%
    group_by(genre) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u)/(n()+lambda))
  # Year Effect (by)
  b_y <- trainset %>% 
    left_join(bi, by ='movieId') %>%
    left_join(bu, by='userId') %>%
    left_join(bg, by= "genre") %>%
    group_by(year) %>%
    summarize(b_y = mean(rating - mu - b_i - b_u - b_g)/(n()+lambda))
  # Prediction: mu + bi + bu + bg + by
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genre") %>%
    left_join(b_y, by = "year") %>%
    filter(!is.na(b_i), !is.na(b_u), !is.na(b_g), !is.na(b_y)) %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

# Define a set of lambdas to tune
lambdas <- seq(0, 10, 0.25)

# Tune lambda
rmses <- sapply(lambdas, 
                regularization, 
                trainset = train_set, 
                testset = test_set)


# Plot the lambda vs RMSE
tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Regularization", 
          subtitle = "Pick the penalization that gives the lowest RMSE.")

####Final Validation####
lambda <- 5.0
mu_edx <- mean(edx$rating)

# Movie effect (bi)
b_i_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect (bu)
b_u_edx <- edx %>% 
  left_join(b_i_edx, by ="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# Genre effect (bg)
b_g_edx <- edx %>% 
  left_join(b_i_edx, by ="movieId") %>%
  left_join(b_u_edx, by ="userId") %>%
  group_by(genre) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu_edx)/(n()+lambda))

# Year effect (by)
b_y_edx <- edx %>% 
  left_join(b_i_edx, by ="movieId") %>%
  left_join(b_u_edx, by ="userId") %>%
  left_join(b_g_edx, by = "genre") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - b_i - b_u - b_g - mu_edx)/(n()+lambda))

# Prediction
y_hat_edx <- validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  left_join(b_g_edx, by = "genre") %>%
  left_join(b_y_edx, by = "year") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g + b_y) %>%
  .$pred

####Validation set arrangements####
validation$userId <- as.character(validation$userId)
validation$movieId <- as.character(validation$movieId)
validation$timestamp <- as_datetime(validation$timestamp, origin = "1970-01-01")
validation <- validation %>% separate(genres, c("genre", "genre2"), sep = "[|]", fill = "warn") %>% select(-genre2)
validation$genre <- as.factor(validation$genre)
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# Final Prediction
Final_result <-  RMSE(validation$rating, y_hat_edx)
Final_result 








