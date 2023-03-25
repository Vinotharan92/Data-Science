#QUEST Decision Tree

cat("\014")  # clear console
rm(list = ls())  # clear environment
graphics.off()  # clear plots tab

#Load required packages
library(party)
library(caret)
library(ROCR)

# read SPSS data into R as a data frame
data <- read.spss("/Users/vinotharanannarasa/BUSN9049/Assignment 1/credit-card-clients.sav", to.data.frame = TRUE)

###############################################Identify and deal with duplicates
data <- unique(data) #I got rid of three observation here

#######################################################Check for missing values
sum(is.na(data)) #No na's
data <- data[complete.cases(data), ]

#######################################################Inspect data
summary(data) #Nothing new identified here
str(data)

###################################Convert nominal and ordinal data to  factors
data$X2 <- factor(data$X2, levels = c(1, 2), labels = c("M", "F"))
data$X3 <- factor(data$X3)
data$X4 <- factor(data$X4)
data$X6 <- factor(data$X6)
data$X7 <- factor(data$X7)
data$X8 <- factor(data$X8)
data$X9 <- factor(data$X9)
data$X10 <- factor(data$X10)
data$X11 <- factor(data$X11)
data$Y <- factor(data$Y, levels = c(0, 1), labels = c("No", "Yes"))

str(data)
##########################################Split data into training and test sets
set.seed(123)
train_index <- createDataPartition(data$Y, p = 0.7, list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]

########################Normalization - Train###################################
data_subset <- train %>% select(X1,X12:X23)
# Scale the selected variables
data_scaled <- scale(data_subset)
train_data_normalized <- cbind(data_scaled, train[, c("X2", paste0("X", 3:11))], train$Y)
train_data_normalized <- rename(train_data_normalized, Y = 'train$Y')

########################Normalization - Test###################################
data_subset <- test %>% select(X1,X12:X23)
# Scale the selected variables
data_scaled <- scale(data_subset)
test_data_normalized <- cbind(data_scaled, test[, c("X2", paste0("X", 3:11))], test$Y)
test_data_normalized <- rename(test_data_normalized, Y = 'test$Y')

################################
# Define the train control and tuning parameters
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE)

# Train the model using the train function and cross-validation
mincriterion_values <- seq(0.4, 0.7, by = 0.01)
quest_model <- train(Y ~ ., data = train_data_normalized, method = "ctree", trControl = train_control, tuneGrid = data.frame(mincriterion = mincriterion_values))
quest_model

#Make predictions on the test set using the trained model

quest_pred <- predict(quest_model, newdata = test_data_normalized, type = "raw")
quest_prob <- predict(quest_model, newdata = test_data_normalized, type = "prob")

#Evaluate model performance

conf_mat <- confusionMatrix(table(test_data_normalized$Y, quest_pred))

#Compute AUC-ROC
roc <- prediction(quest_prob[,2], test_data_normalized$Y)
auc <- as.numeric(performance(roc, "auc")@y.values)
auc #0.7546629

# extract performance metrics from the confusion matrix
accuracy <- conf_mat$overall["Accuracy"]
sensitivity <- conf_mat$byClass["Sensitivity"]
specificity <- conf_mat$byClass["Specificity"]
precision <- conf_mat$byClass["Pos Pred Value"]
recall <- sensitivity  # same as sensitivity

# extract performance metrics from the confusion matrix
accuracy <- conf_mat$overall["Accuracy"]
sensitivity <- conf_mat$byClass["Sensitivity"]
specificity <- conf_mat$byClass["Specificity"]
precision <- conf_mat$byClass["Pos Pred Value"]
recall <- sensitivity  # same as sensitivity

# print the performance metrics
cat(sprintf("Accuracy: %0.3f\n", accuracy)) #0.822
cat(sprintf("Sensitivity: %0.3f\n", sensitivity)) #0.839
cat(sprintf("Specificity: %0.3f\n", specificity)) #0.687
cat(sprintf("Precision: %0.3f\n", precision)) #0.953
cat(sprintf("Recall: %0.3f\n", recall)) #0.839

#Find optimal threshold using Youden's J statistic
perf <- performance(roc, "tpr", "fpr")
coords <- cbind(unlist(perf@x.values), unlist(perf@y.values))
j_stat <- coords[,2] - coords[,1]
opt_idx <- which.max(j_stat)
opt_threshold <- cbind(coords[opt_idx,1], coords[opt_idx,2])

#Plot ROC curve and optimal coordinate

plot(perf, col="blue")
abline(a=0, b=1, lty=2, col="gray")
points(opt_threshold, col="red", p=19, cex=1.5)

threshold_value = unlist(perf@alpha.values)[opt_idx] #Get the optimal value 0.3402062

quest_pred <- ifelse(quest_prob[,2] > threshold_value, "Yes", "No")
