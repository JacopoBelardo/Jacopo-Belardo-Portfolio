# BOSTON HOUSING MEDIAN HOME PRICES

options(scipen=999, digits = 5)

library(caret)
hous.df <- mlba::BostonHousing
View(hous.df)
str(hous.df)
summary(hous.df)

# Convert CHAS and CAT.MEDV to factors
hous.df$CHAS <- as.factor(hous.df$CHAS)
hous.df$CAT.MEDV <- as.factor(hous.df$CAT.MEDV)

str(hous.df)

# a. Why should the data be partitioned into training and holdout sets?
#   `What will the training set be used for?
#    What will the holdout set be used for?`

# The data should be partitioned to train the model with a large number of observations and then test its predictive power on a smaller amount of data. 
# The training set is used to fit the model and help it learn patterns and relationships between variables. 
# The holdout set is used to evaluate how well the model generalizes to unseen data by ensuring it doesn't just perform well on the data it was trained on.

# partition data
set.seed(1)
idx <- createDataPartition(hous.df$MEDV, p=0.6, list=FALSE)
train.df <- hous.df[idx, ]
holdout.df <- hous.df[-idx, ]


# b. Fit a multiple linear regression to the median house price as a function of CRIM, CHAS, and RM. 

hous.lm <- lm(MEDV ~ CRIM + CHAS + RM, data=train.df)
summary(hous.lm)

# MEDV = −30.7368 − 0.2613×CRIM + 3.5746×CHAS + 8.5627×RM

# c. Using the estimated regression model, what median house is predicted for a tract in the Boston area 
#    that does not bound the Charles River, has a crime rate of 0.1, and where the average number of rooms per house is 6?

# Create a new data frame with the values for CRIM, CHAS, and RM
new.data <- data.frame(CRIM = 0.1, CHAS = "0", RM = 6)

# Use the predict function to calculate the predicted MEDV
predicted_value <- predict(hous.lm, newdata = new.data)

predicted_value

# The predicted price for the house is $20,613

# d. Reduce the number of predictors:

#   i. Which predictors are likely to be measuring the same thing among the 13 predictors? 
#      Discuss the relationships among INDUS, NOX, and TAX.

# Looking at the description of the variables, I noticed that there are some predictors giving us similar information.
# INDUS, NOX, and TAX all seem to describe a property located in a very industrialized area influenced by high pollution levels.
# INDUS measures the proportion of industrial acres.
# NOX measures the nitric oxide concentration.
# TAX represents the tax-rate for the area.
# INDUS - NOX: these two variables probably are positively correlated because nitric oxide concentration is usually higher in urbanized areas with many factories or offices.
# INDUS - TAX: also in this case there might be positive correlation because industrialized areas tend to have higher taxes due to the presence of many buildings and infrastructures.
# NOX - TAX: lastly, these two variables might be positively correlated because it is a common practice to raise tax rates in industrialized areas to manage the impact of the pollution.

# We can check correlations matrix to have an empirical proof of my hypothesis.
cor(hous.df[, c("INDUS", "NOX", "TAX")])
# The results confirms that there is a substantial positive correlation between the variables.

#   ii. Compute the correlation table for the 12 numerical predictors, and search for highly correlated pairs.
#       These have potential redundancy and can cause multicollinearity.
#       Choose which ones to remove based on this table.

# Exclude CHAS (categorical) and CAT.MEDV (categorical)
numerical_predictors <- hous.df[, c("CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT", "MEDV")]

# Compute correlation matrix
cor(numerical_predictors)

# Variables with strong correlation:
# INDUS - NOX (positive)
# INDUS - TAX (positive)
# NOX - AGE (positive)
# NOX - DIS (negative)
# AGE - DIS (negative)
# RAD - TAX (positive)

# Based on these findings I decided to remove the following variables: NOX, TAX, DIS

library(dplyr)
# Create a new dataframe excluding NOX, TAX, and DIS (and also CAT.MEDV because I am not going to use it)
new_hous.df <- hous.df %>%
  select(-NOX, -TAX, -DIS, -CAT.MEDV)

View(new_hous.df)
# Check the structure of the new dataframe
str(new_hous.df)

# Calculate the correlation matrix excluding CHAS
correlation_matrix <- new_hous.df %>%
  select(-CHAS) %>%
  cor()
print(correlation_matrix)


#   iii. Use stepwise regression with the three options (backward, forward, and both) to reduce the remaining predictors as follows:
#        Run stepwise on the training set.
#        Choose the top model from each stepwise run.
#        Then use each of the models separately to predict the holdout set.
#        Compare RMSE and mean absolute error, as well as lift charts.
#        Finally, describe the best model.

library(mlba)

# partition data
set.seed(1)
idx <- createDataPartition(new_hous.df$MEDV, p=0.6, list=FALSE)
train.df <- new_hous.df[idx, ]
holdout.df <- new_hous.df[-idx, ]

trControl <- caret::trainControl(method = "none")

# Backward elimination
backward_model <- caret::train(MEDV ~ ., data = train.df, trControl = trControl,
                               method = "glmStepAIC", direction = 'backward')

coef(backward_model$finalModel)

# MEDV = 12.98831 − 0.11189×CRIM + 3.43211×CHAS1 + 4.67053×RM + 0.07358×RAD − 0.68262×PTRATIO − 0.63905×LSTAT

# Forward selection
forward_model <- caret::train(MEDV ~ ., data = train.df, trControl = trControl,
                              method = "glmStepAIC", direction = 'forward')

coef(forward_model$finalModel)

# MEDV = 12.98831 − 0.63905×LSTAT + 4.67053×RM − 0.68262×PTRATIO + 3.43211×CHAS1 − 0.11189×CRIM + 0.07358×RAD

# Both directions
both_model <- caret::train(MEDV ~ ., data = train.df, trControl = trControl,
                           method = "glmStepAIC", direction = 'both')

coef(both_model$finalModel)

# MEDV = 12.98831 − 0.11189×CRIM + 3.43211×CHAS1 + 4.67053×RM + 0.07358×RAD − 0.68262×PTRATIO − 0.63905×LSTAT

# Make predictions on the holdout set for each model

# Make predictions on the holdout set for each model
# Backward Model Predictions
backward_predictions <- predict(backward_model, holdout.df)

# Forward Model Predictions
forward_predictions <- predict(forward_model, holdout.df)

# Both Directions Model Predictions
both_predictions <- predict(both_model, holdout.df)


# Summary for Backward Model
backward_summary <- rbind(
  Training = mlba::regressionSummary(predict(backward_model, train.df), train.df$MEDV),
  Holdout = mlba::regressionSummary(predict(backward_model, holdout.df), holdout.df$MEDV)
)

# Summary for Forward Model
forward_summary <- rbind(
  Training = mlba::regressionSummary(predict(forward_model, train.df), train.df$MEDV),
  Holdout = mlba::regressionSummary(predict(forward_model, holdout.df), holdout.df$MEDV)
)

# Summary for Both Model
both_summary <- rbind(
  Training = mlba::regressionSummary(predict(both_model, train.df), train.df$MEDV),
  Holdout = mlba::regressionSummary(predict(both_model, holdout.df), holdout.df$MEDV)
)


backward_summary
forward_summary
both_summary

# Compare predictions visually
boxplot(backward_predictions, forward_predictions, both_predictions, 
        names = c("Backward", "Forward", "Both"),
        main = "Comparison of Model Predictions", 
        ylab = "Predicted MEDV")

# Lift charts

library(ggplot2)
library(gridExtra)
library(gains)

# Function to create cumulative gains and lift charts
create_charts <- function(predictions, model_name) { # used ChatGPT advice to make it a function
  price <- holdout.df$MEDV
  gain <- gains(price, predictions)
  
  # Cumulative lift chart
  df_cumulative <- data.frame(
    ncases = c(0, gain$cume.obs),
    cumPrice = c(0, gain$cume.pct.of.total * sum(price))
  )
  
  g1 <- ggplot(df_cumulative, aes(x = ncases, y = cumPrice)) +
    geom_line() +
    geom_line(data = data.frame(ncases = c(0, nrow(holdout.df)), cumPrice = c(0, sum(price))),
              color = "gray", linetype = 2) + # adds baseline
    labs(x = "# Cases", y = "Cumulative Price", title = paste("Cumulative Gains Chart (", model_name, ")", sep="")) +
    scale_y_continuous(labels = scales::comma) 
  
  # Decile-wise lift chart
  df_decile <- data.frame(
    percentile = gain$depth,
    meanResponse = gain$mean.resp / mean(price)
  )
  
  g2 <- ggplot(df_decile, aes(x = percentile, y = meanResponse)) +
    geom_bar(stat = "identity") +
    labs(x = "Percentile", y = "Decile Mean / Global Mean", title = paste("Decile-wise Lift Chart (", model_name, ")", sep=""))
  
  return(list(cumulative_chart = g1, decile_chart = g2))
}

# Create charts for each model
backward_charts <- create_charts(backward_predictions, "Backward Model")
forward_charts <- create_charts(forward_predictions, "Forward Model")
both_charts <- create_charts(both_predictions, "Both Directions Model")

# Combine them all in one plot arrangement
g <- arrangeGrob(
  backward_charts$cumulative_chart + theme_bw(),
  backward_charts$decile_chart + theme_bw() + scale_x_continuous(breaks = seq(10, 100, by = 10)),
  forward_charts$cumulative_chart + theme_bw(),
  forward_charts$decile_chart + theme_bw() + scale_x_continuous(breaks = seq(10, 100, by = 10)),
  both_charts$cumulative_chart + theme_bw(),
  both_charts$decile_chart + theme_bw() + scale_x_continuous(breaks = seq(10, 100, by = 10)),
  ncol = 2
)

# Display combined plots
gridExtra::grid.arrange(g)

# Looking at performance evaluation metrics, the three models are very similar.
# The RMSE and MAE values for all three stepwise regression models 
# are almost identical, both for the training and holdout sets. This indicates that all models have the same predictive performance on the new data. 
# The lift charts are also identical, suggesting no significant difference in their ability to rank predictions. 
# Given this, we can choose any of the models as the "best" since they provide equivalent results.
# To keep things simple, I'll choose the backward elimination model, which tends to be slightly more interpretable
# due to the starting point of including all predictors and progressively removing non-significant ones.

summary(backward_model$finalModel)

# This is the final model I decided to use:
# MEDV = 12.98831 − 0.11189×CRIM + 3.43211×CHAS1 + 4.67053×RM + 0.07358×RAD − 0.68262×PTRATIO − 0.63905×LSTAT
# From this we can draw the following conclusions:
# - Higher crime rate decreases median house prices
# - Proximity to the Charles River increases median house prices
# - Higher number of rooms increase median house prices
# - Access to radial highways slightly increases median house prices
# - Higher pupil-teacher ratio decreases median houses prices
# - Larger percentage of lower status population decreases median house prices