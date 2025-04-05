
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
data <- read.csv("/Users/ayush/Documents/Math 4322/Project/AirplaneDelays_MachineLearning/Dataset/Jan_2020_ontime.csv")

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

set.seed(123)
sample_index <- sample(1:nrow(flight_data), size = 0.5 * nrow(flight_data))
sampled_data <- flight_data[sample_index, ]

# Split into training (75%) and test set (25%)
set.seed(42)
trainIndex <- createDataPartition(sampled_data$DEP_DEL15, p = 0.75, list = FALSE)
train_data <- flight_data[trainIndex, ]
test_data <- flight_data[-trainIndex, ]

# balanced_train <- ovun.sample(DEP_DEL15 ~ ., data = train_data, method = "both", p = 0.5, seed = 1)$data
balanced_train_smote <- SMOTE(DEP_DEL15 ~ ., data = train_data, perc.over = 100, perc.under = 200)


# Logistic Regression Model
logistic_model <- glm(DEP_DEL15 ~ ., data = balanced_train, family = binomial)
summary(logistic_model)

# Predict on test data
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)
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

