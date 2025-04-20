
install.packages("rpart")         # For Decision Tree
install.packages("randomForest")  # For Random Forest
install.packages("caret")
install.packages("pROC")
install.packages("ROSE")
install.packages("lubridate")
install.packages("ggplot2")
install.packages("DMwR2")
install.packages("doParallel")

library(lubridate)
library(ggplot2)

library(rpart)          # For Decision Tree
library(randomForest)   # For Random Forest

library(ggplot2)
library(caret)
library(pROC)
library(ROSE)


# head(Jan_2019_ontime)

# Load Dataset
# data <- read.csv("/Users/ayush/Documents/Math 4322/Project/AirplaneDelays_MachineLearning/Dataset/Jan_2019_ontime.csv")
data <- Jan_2020_ontime

# Select relevant features
flight_data <- data[c("DAY_OF_WEEK", "OP_CARRIER", "ORIGIN", "DEST", "DEP_TIME", "DISTANCE", "DEP_DEL15")]

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
flight_data$DAY_OF_WEEK <- as.factor(flight_data$DAY_OF_WEEK)
flight_data$OP_CARRIER <- as.factor(flight_data$OP_CARRIER)
flight_data$ORIGIN <- as.factor(flight_data$ORIGIN)
flight_data$DEST <- as.factor(flight_data$DEST)
flight_data$DEP_DEL15 <- as.factor(flight_data$DEP_DEL15)

set.seed(42)
sample_index <- sample(1:nrow(flight_data), size = 0.5 * nrow(flight_data))
sampled_data <- flight_data[sample_index, ]

# Split into training (75%) and test set (25%)
set.seed(42)
trainIndex <- createDataPartition(sampled_data$DEP_DEL15, p = 0.75, list = FALSE)
train_data <- flight_data[trainIndex, ]
test_data <- flight_data[-trainIndex, ]

# balanced_train <- ovun.sample(DEP_DEL15 ~ ., data = train_data, method = "both", p = 0.5, seed = 1)$data

# balanced_train_smote <- SMOTE(DEP_DEL15 ~ ., data = train_data, perc.over = 100, perc.under = 200)
# balanced_train <- ROSE(DEP_DEL15 ~ ., data = train_data, seed = 1)$data

library(caret)
library(DMwR2)  # for SMOTE if needed

# Set up training control with ROSE or SMOTE
ctrl <- trainControl(
  method = "cv",              # 5-fold cross-validation
  number = 7,
  sampling = "rose",          # or use "smote"
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# balanced_train_ub <- ubBalance(train_data, method = "smote", percOver = 100, percUnder = 200)


# Logistic Regression Model
logistic_model <- glm(DEP_DEL15 ~ ., data = balanced_train, family = binomial)


summary(logistic_model)

# Predict on test data
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, "1", "0", levels = c("0", "1"))
logistic_pred_class <- as.factor(logistic_pred_class)

# Confusion Matrix
confusionMatrix(logistic_pred_class, test_data$DEP_DEL15)

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




# Random Forest model
# rf_model <- randomForest(DEP_DEL15 ~ ., data = balanced_train, ntree = 300, mtry = 3, importance = TRUE)

# Rename factor levels to valid names
train_data$DEP_DEL15 <- factor(train_data$DEP_DEL15, levels = c("0", "1"), labels = c("No", "Yes"))
test_data$DEP_DEL15 <- factor(test_data$DEP_DEL15, levels = c("0", "1"), labels = c("No", "Yes"))

library(doParallel)

# Use all but 1 core
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)


set.seed(42)
rf_model_caret <- train(DEP_DEL15 ~ ., data = train_data,
                        method = "rf",
                        trControl = ctrl,
                        ntree = 50,
                        importance = TRUE)

stopCluster(cl)

getDoParWorkers()

registerDoSEQ()  


rf_pred_class <- predict(rf_model_caret, newdata = test_data)
rf_pred_prob <- predict(rf_model_caret, newdata = test_data, type = "prob")

confusionMatrix(rf_pred_class, test_data$DEP_DEL15, positive = "Yes")






varImpPlot(rf_model)

# Predict on test data
# rf_pred <- predict(rf_model_caret, newdata = test_data, type = "prob")
# rf_pred_class <- ifelse(rf_pred_prob[,"Yes"] > 0.2, "Yes", "No")
# rf_pred_class <- factor(rf_pred_class, levels = c("No", "Yes"))
# confusionMatrix(rf_pred_class, test_data$DEP_DEL15, positive = "Yes")




thresholds <- seq(0.1, 0.9, by = 0.05)
f1_scores <- c()

for (thresh in thresholds) {
  pred_class <- ifelse(rf_pred_prob[,"Yes"] > thresh, "Yes", "No")
  pred_class <- factor(pred_class, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred_class, test_data$DEP_DEL15, positive = "Yes")
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  f1_scores <- c(f1_scores, f1)
}

# Best threshold
best_thresh <- thresholds[which.max(f1_scores)]
cat("📌 Best Threshold (F1):", best_thresh, "with F1 Score:", max(f1_scores), "\n")




# saveRDS(rf_model_caret, "rf_delay_model.rds")



# Create a data frame with one random (but valid) observation
new_flight <- data.frame(
  DAY_OF_WEEK = factor("3", levels = levels(train_data$DAY_OF_WEEK)),
  OP_CARRIER = factor("OTHER", levels = levels(train_data$OP_CARRIER)),
  ORIGIN = factor("DEN", levels = levels(train_data$ORIGIN)),
  DEST = factor("ATL", levels = levels(train_data$DEST)),
  DEP_TIME = 1610,
  DISTANCE = 1119
)


predict_delay <- function(model, newdata, threshold = 0.2) {
  probs <- predict(model, newdata = newdata, type = "prob")
  pred <- ifelse(probs[,"Yes"] > threshold, "Yes", "No")
  return(factor(pred, levels = c("No", "Yes")))
}

# Predict
pred_result <- predict_delay(rf_model_caret, new_flight, threshold = 0.2)
print(paste("Prediction:", pred_result))


levels(train_data$DEST)


