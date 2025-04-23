
install.packages("rpart")         # For Decision Tree
install.packages("randomForest")  # For Random Forest
install.packages("caret")
install.packages("pROC")
install.packages("ROSE")
install.packages("lubridate")
install.packages("ggplot2")
install.packages("DMwR2")

library(lubridate)
library(ggplot2)

library(rpart)          # For Decision Tree
library(randomForest)   # For Random Forest

library(ggplot2)
library(caret)
library(pROC)
library(ROSE)
library(DMwR2)



head(Jan_2020_ontime)

# Load Dataset
data <- read.csv("/Users/ayush/Documents/Math 4322/Project/AirplaneDelays_MachineLearning/Dataset/Jan_2019_ontime.csv")

# Select relevant features
flight_data <- data[c("DAY_OF_WEEK", "DAY_OF_MONTH", "OP_CARRIER", "ORIGIN", 
                              "DEST", "DEP_TIME", "DEP_TIME_BLK", "DISTANCE", "DEP_DEL15")]

# Remove rows with NA
flight_data <- na.omit(flight_data)

# Reduce factor levels to top 10 only for performance
top_origin <- names(sort(table(flight_data$ORIGIN), decreasing = TRUE)[1:10])
flight_data$ORIGIN <- ifelse(flight_data$ORIGIN %in% top_origin, flight_data$ORIGIN, "OTHER")

top_dest <- names(sort(table(flight_data$DEST), decreasing = TRUE)[1:10])
flight_data$DEST <- ifelse(flight_data$DEST %in% top_dest, flight_data$DEST, "OTHER")

top_carrier <- names(sort(table(flight_data$OP_CARRIER), decreasing = TRUE)[1:10])
flight_data$OP_CARRIER <- ifelse(flight_data$OP_CARRIER %in% top_carrier, flight_data$OP_CARRIER, "OTHER")



# Convert categorical variables to factors
# flight_data$DAY_OF_WEEK <- as.factor(flight_data$DAY_OF_WEEK)
# flight_data$OP_CARRIER <- as.factor(flight_data$OP_CARRIER)
# flight_data$ORIGIN <- as.factor(flight_data$ORIGIN)
# flight_data$DEST <- as.factor(flight_data$DEST)
# flight_data$DEP_DEL15 <- as.factor(flight_data$DEP_DEL15)


flight_data$DAY_OF_WEEK <- as.factor(flight_data$DAY_OF_WEEK)
flight_data$DAY_OF_MONTH <- as.factor(flight_data$DAY_OF_MONTH)
flight_data$OP_CARRIER <- as.factor(flight_data$OP_CARRIER)
flight_data$ORIGIN <- as.factor(flight_data$ORIGIN)
flight_data$DEST <- as.factor(flight_data$DEST)
flight_data$DEP_TIME_BLK <- as.factor(flight_data$DEP_TIME_BLK)
flight_data$DEP_DEL15 <- as.factor(flight_data$DEP_DEL15)


set.seed(123)
sample_index <- sample(1:nrow(flight_data), size = 0.5 * nrow(flight_data))
sampled_data <- flight_data[sample_index, ]

# Split into training (75%) and test set (25%)
# set.seed(42)
# trainIndex <- createDataPartition(sampled_data$DEP_DEL15, p = 0.75, list = FALSE)
# train_data <- flight_data[trainIndex, ]
# test_data <- flight_data[-trainIndex, ]

library(caret)
set.seed(42)
trainIndex <- createDataPartition(flight_data$DEP_DEL15, p = 0.80, list = FALSE)
train_data <- flight_data[trainIndex, ]
test_data <- flight_data[-trainIndex, ]

# balanced_train <- ovun.sample(DEP_DEL15 ~ ., data = train_data, method = "both", p = 0.5, seed = 1)$data
# balanced_train_smote <- SMOTE(DEP_DEL15 ~ ., data = train_data, perc.over = 100, perc.under = 200)

library(ROSE)
balanced_train <- ROSE(DEP_DEL15 ~ ., data = train_data, seed = 123)$data

# Logistic Regression Model
# logistic_model <- glm(DEP_DEL15 ~ ., data = balanced_train, family = binomial)
# summary(logistic_model)
# 
# # Predict on test data
# logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
# logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)
# logistic_pred_class <- as.factor(logistic_pred_class)

logistic_model <- glm(DEP_DEL15 ~ ., data = balanced_train, family = binomial)
summary(logistic_model)

# Splitting data into training/testing and running Logistic Regression 
# with selected predictors over 10 iterations

set.seed(42)
lg.error.rate <- numeric(10)

for (i in 1:10) {
  sample <- sample.int(n = nrow(flight_data), size = round(0.80 * nrow(flight_data)), replace = FALSE)
  train <- flight_data[sample, ]
  test <- flight_data[-sample, ]
  
  flight.glm <- glm(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN + 
                      DEST + DEP_TIME_BLK + DISTANCE,
                    data = train, family = "binomial")
  
  # To obtain error rate
  glm.pred <- predict(flight.glm, newdata = test, type = "response")
  yHat <- glm.pred > 0.5
  lg.cm <- table(test$DEP_DEL15, yHat)
  lg.error.rate[i] <- ((lg.cm[1,2] + lg.cm[2,1]) / sum(lg.cm))
  lg.error.rate[i]
}

# Output the mean test error
mean(lg.error.rate)

# Predict on test data
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)
logistic_pred_class <- as.factor(logistic_pred_class)

# Confusion Matrix 
conf_matrix <- confusionMatrix(logistic_pred_class, test_data$DEP_DEL15)
print(conf_matrix)


library(tree)
# Load library
library(ROSE)

# Ensure target variable is a factor
train_data$DEP_DEL15 <- as.factor(train_data$DEP_DEL15)

# Apply ROSE to generate a balanced dataset
set.seed(42)  # for reproducibility
train_rose <- ROSE(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN + 
                     DEST + DEP_TIME_BLK + DISTANCE, 
                   data = train_data, seed = 42)$data

# Check class balance
table(train_rose$DEP_DEL15)

# Train the decision tree on the balanced ROSE data
tree_model <- tree(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN + 
                     DEST + DEP_TIME_BLK + DISTANCE, 
                   data = train_rose,
                   control = tree.control(nobs = nrow(train_rose), mindev = 0.001))

# Print the model summary
summary(tree_model)

# Set up storage for error rates
set.seed(42)
tree_error_rate <- numeric(10)

for (i in 1:10) {
  # Random 80/20 split
  sample <- sample.int(n = nrow(flight_data), size = round(0.80 * nrow(flight_data)), replace = FALSE)
  train <- flight_data[sample, ]
  test <- flight_data[-sample, ]
  
  # Train decision tree
  tree_model_iter <- tree(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN + 
                            DEST + DEP_TIME_BLK + DISTANCE,
                          data = train,
                          control = tree.control(nobs = nrow(train), mindev = 0.001))
  
  # Predict
  tree_pred_iter <- predict(tree_model_iter, newdata = test, type = "class")
  
  # Compute confusion matrix and error rate
  cm <- table(test$DEP_DEL15, tree_pred_iter)
  error <- sum(cm[row(cm) != col(cm)]) / sum(cm)
  tree_error_rate[i] <- error
}

# Mean test error across 10 iterations
mean(tree_error_rate)



# Make predictions on the test set
tree_pred <- predict(tree_model, newdata = test_data, type = "class")

# Confusion matrix
tree_conf_matrix <- confusionMatrix(tree_pred, test_data$DEP_DEL15)
print(tree_conf_matrix)

# Performance metrics
tree_accuracy <- tree_conf_matrix$overall["Accuracy"]
tree_sensitivity <- tree_conf_matrix$byClass["Sensitivity"]
tree_specificity <- tree_conf_matrix$byClass["Specificity"]

cat("Decision Tree Performance (tree package):\n")
cat("Accuracy:", tree_accuracy, "\n")
cat("Sensitivity:", tree_sensitivity, "\n")
cat("Specificity:", tree_specificity, "\n")



# Prune the tree using cross-validation
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
plot(cv_tree$size, cv_tree$dev, type = "b", 
     xlab = "Tree Size", ylab = "Misclassification Rate", main = "CV for Tree Pruning")

# Select optimal size
optimal_size <- cv_tree$size[which.min(cv_tree$dev)]
cat("Optimal tree size:", optimal_size, "\n")

# Prune the tree
pruned_tree <- prune.misclass(tree_model, best = optimal_size)
# Visualize the pruned tree
plot(pruned_tree, type = "uniform")  
text(pruned_tree, pretty = 0, cex = 0.5) 

# Predictions using pruned tree
pruned_pred <- predict(pruned_tree, newdata = test_data, type = "class")

# Confusion matrix for pruned tree
pruned_conf_matrix <- confusionMatrix(pruned_pred, test_data$DEP_DEL15)
print(pruned_conf_matrix)

# Compare accuracy before and after pruning
cat("Comparison of accuracy before and after pruning:\n")
cat("Original tree accuracy:", tree_accuracy, "\n")
cat("Pruned tree accuracy:", pruned_conf_matrix$overall["Accuracy"], "\n")



# Visualize the decision tree
plot(tree_model, type = "uniform")  
text(tree_model, pretty = 0, cex = 0.5) 



# Load required library
library(randomForest)

# Train the random forest model
set.seed(42)
rf_model <- randomForest(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + 
                           ORIGIN + DEST + DEP_TIME_BLK + DISTANCE,
                         data = balanced_train, ntree = 100, importance = TRUE)

# Print model summary
print(rf_model)

# === Random Forest 10-Iteration Evaluation ===

library(randomForest)

set.seed(42)
rf.error.rate <- numeric(10)

for (i in 1:10) {
  # Random 80/20 train-test split
  sample <- sample.int(n = nrow(flight_data), size = round(0.80 * nrow(flight_data)), replace = FALSE)
  train <- flight_data[sample, ]
  test <- flight_data[-sample, ]
  
  # Balance the training set using ROSE
  train_bal <- ROSE(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN +
                      DEST + DEP_TIME_BLK + DISTANCE, data = train, seed = 42)$data
  
  # Train Random Forest
  rf_model_iter <- randomForest(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN +
                                  DEST + DEP_TIME_BLK + DISTANCE,
                                data = train_bal, ntree = 100)
  
  # Predict
  rf_pred <- predict(rf_model_iter, newdata = test)
  
  # Confusion matrix and error rate
  cm <- table(test$DEP_DEL15, rf_pred)
  error <- sum(cm[row(cm) != col(cm)]) / sum(cm)
  rf.error.rate[i] <- error
}

# Mean test error across 10 iterations
mean(rf.error.rate)


# Variable importance plot
varImpPlot(rf_model, main = "Variable Importance - Random Forest")



# Make predictions on test data
rf_pred <- predict(rf_model, newdata = test_data)

# Confusion matrix
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$DEP_DEL15)
print(rf_conf_matrix)

# Extract metrics
rf_accuracy <- rf_conf_matrix$overall["Accuracy"]
rf_sensitivity <- rf_conf_matrix$byClass["Sensitivity"]
rf_specificity <- rf_conf_matrix$byClass["Specificity"]

cat("Random Forest Performance:\n")
cat("Accuracy:", rf_accuracy, "\n")
cat("Sensitivity (True Positive Rate):", rf_sensitivity, "\n")
cat("Specificity (True Negative Rate):", rf_specificity, "\n")



# === Feature Engineering ===
balanced_train$DEP_HOUR <- floor(as.numeric(as.character(balanced_train$DEP_TIME)) / 100)
test_data$DEP_HOUR <- floor(as.numeric(as.character(test_data$DEP_TIME)) / 100)

balanced_train$IS_WEEKEND <- ifelse(balanced_train$DAY_OF_WEEK %in% c("6", "7"), 1, 0)
test_data$IS_WEEKEND <- ifelse(test_data$DAY_OF_WEEK %in% c("6", "7"), 1, 0)

balanced_train$ROUTE <- paste(balanced_train$ORIGIN, balanced_train$DEST, sep = "_")
test_data$ROUTE <- paste(test_data$ORIGIN, test_data$DEST, sep = "_")

# === Limit ROUTE to Top 50 Most Common Routes ===
top_routes <- names(sort(table(balanced_train$ROUTE), decreasing = TRUE))[1:50]
balanced_train$ROUTE <- ifelse(balanced_train$ROUTE %in% top_routes, balanced_train$ROUTE, "OTHER")
test_data$ROUTE <- ifelse(test_data$ROUTE %in% top_routes, test_data$ROUTE, "OTHER")

# === Convert to Factors ===
balanced_train$DEP_HOUR <- as.factor(balanced_train$DEP_HOUR)
test_data$DEP_HOUR <- as.factor(test_data$DEP_HOUR)

balanced_train$IS_WEEKEND <- as.factor(balanced_train$IS_WEEKEND)
test_data$IS_WEEKEND <- as.factor(test_data$IS_WEEKEND)

balanced_train$ROUTE <- factor(balanced_train$ROUTE, levels = unique(balanced_train$ROUTE))
test_data$ROUTE <- factor(test_data$ROUTE, levels = levels(balanced_train$ROUTE))

# === Ensure All Factor Levels Match Between Train/Test ===
factor_columns <- c("DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER", "ORIGIN", 
                    "DEST", "DEP_TIME_BLK", "DEP_HOUR", "IS_WEEKEND", "ROUTE")

for (col in factor_columns) {
  test_data[[col]] <- factor(test_data[[col]], levels = levels(balanced_train[[col]]))
}

# === Train Random Forest Model ===
set.seed(42)
rf_model <- randomForest(DEP_DEL15 ~ DAY_OF_MONTH + DAY_OF_WEEK + OP_CARRIER + ORIGIN + 
                           DEST + DEP_TIME_BLK + DISTANCE + DEP_HOUR + IS_WEEKEND + ROUTE,
                         data = balanced_train,
                         ntree = 200, mtry = 3, nodesize = 5, importance = TRUE)

# === Make Predictions ===
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")

# Adjust threshold to favor predicting delays
rf_pred_adjusted <- ifelse(rf_probs[, "1"] > .5, "1", "0")
rf_pred_adjusted <- factor(rf_pred_adjusted, levels = c("0", "1"))

# === Evaluate Performance ===
rf_conf_matrix <- confusionMatrix(rf_pred_adjusted, test_data$DEP_DEL15)
print(rf_conf_matrix)

cat("\nRandom Forest Performance:\n")
cat("Accuracy:", rf_conf_matrix$overall["Accuracy"], "\n")
cat("Sensitivity:", rf_conf_matrix$byClass["Sensitivity"], "\n")
cat("Specificity:", rf_conf_matrix$byClass["Specificity"], "\n")
cat("NPV:", rf_conf_matrix$byClass["Neg Pred Value"], "\n")

varImpPlot(rf_model, main = "Variable Importance - Enhanced Random Forest")



# ROC Curve
roc_log <- roc(as.numeric(test_data$DEP_DEL15), as.numeric(logistic_pred))
plot(roc_log, col = "darkgreen", main = "ROC Curve - Logistic Regression")

# Visualization: Predicted Probabilities vs. Actual Labels
test_data$predicted_prob <- logistic_pred
test_data$actual <- test_data$DEP_DEL15

ggplot(test_data, aes(x = predicted_prob, fill = actual)) +
  geom_histogram(binwidth = 0.05, position = "identity", alpha = 0.6) +
  labs(title = "Logistic Regression Predicted Probabilities",
       x = "Predicted Probability of Delay",
       y = "Count") +
  theme_minimal()


models <- c("Logistic Regression", "Decision Tree", "Random Forest")
accuracy <- c(0.6845, 0.829, 0.6935)
sensitivity <- c(0.6803, 0.9931, 0.7123)
specificity <- c(0.7043, 0.0514, 0.6040)

barplot_matrix <- rbind(accuracy, sensitivity, specificity)

barplot(barplot_matrix, beside = TRUE, col = c("skyblue", "salmon", "palegreen"),
        names.arg = models,
        ylim = c(0, 1), main = "Model Performance Comparison",
        ylab = "Metric Score")
legend("topright", legend = rownames(barplot_matrix),
       fill = c("skyblue", "salmon", "palegreen"))
