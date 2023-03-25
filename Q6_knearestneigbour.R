# load required packages
library(foreign) # for reading SPSS data
library(dplyr)
library(caret)
library(ROCR)

# read SPSS data into R as a data frame
data <- read.spss("/Users/vinotharanannarasa/BUSN9049/Assignment 1/credit-card-clients.sav", to.data.frame = TRUE)

###############################################Identify and deal with duplicates
data <- unique(data) #I got rid of three observation here

#######################################################Check for missing values
sum(is.na(data)) #No na's
data <- data[complete.cases(data), ]

#######################################################Inspect Data
summary(data) #Nothing new identified here
str(data)

table(data$X1) #Anount of credit given - Numeric

table(data$X2) #Gender? - Nominal
data$X2 <- factor(data$X2, levels = c(1, 2), labels = c("M", "F"))
table(data$X3) #Education? - Nominal
data$X3 <- factor(data$X3)
table(data$X4) #Marital status? - Nominal
data$X4 <- factor(data$X4)

table(data$X5) #Age - Numeric

table(data$X6) #the repayment status in September, 2005 - Ordinal
data$X6 <- factor(data$X6)
table(data$X7) #the repayment status in August, 2005 - Ordinal
data$X7 <- factor(data$X7)
table(data$X8) #the repayment status in July, 2005 - Ordinal
data$X8 <- factor(data$X8)
table(data$X9) #the repayment status in June, 2005 - Ordinal
data$X9 <- factor(data$X9)
table(data$X10) #the repayment status in May, 2005 - Ordinal
data$X10 <- factor(data$X10)
table(data$X11) #the repayment status in April, 2005 - Ordinal
data$X11 <- factor(data$X11)

table(data$X12) #amount of bill statement in September, 2005 - Numeric
table(data$X13) #amount of bill statement in August, 2005 - Numeric
table(data$X14) #amount of bill statement in July, 2005 - Numeric
table(data$X15) #amount of bill statement in June, 2005 - Numeric
table(data$X16) #amount of bill statement in May, 2005 - Numeric
table(data$X17) #amount of bill statement in April, 2005 - Numeric

table(data$X18) #amount paid in September, 2005 - Numeric
table(data$X19) #amount paid in August, 2005 - Numeric
table(data$X20) #amount paid in July, 2005 - Numeric
table(data$X21) #amount paid in June, 2005 - Numeric
table(data$X22) #amount paid in May, 2005 - Numeric
table(data$X23) #amount paid in April, 2005 - Numeric
###############################################################################
#Convert Y to a factor
data$Y <- factor(data$Y, levels = c(0, 1), labels = c("No", "Yes"))
###############################################################################
# Split data into training and test sets
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

############################# Find the optimal k value##########################
set.seed(123)
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE)
k_values <- seq(1, 50, by = 1)
knn_model <- train(Y ~ ., data = train_data_normalized, method = "knn", trControl = train_control, tuneGrid = data.frame(k = k_values))
knn_model

##################MAKE PREDICTION ON THE TEST SET USING OPTIMAL K###############
knn_pred <- predict(knn_model, newdata = test_data_normalized) # Make predictions on test set using the optimal k value
knn_prob <- predict(knn_model, newdata = test_data_normalized,type = "prob") # Make predictions on test set using the optimal k value

conf_mat <- confusionMatrix(table(test_data_normalized$Y,knn_pred)) #Evaluate model performance
# Compute AUC-ROC
roc <- prediction(knn_prob[,2], test_data_normalized$Y)
auc <- as.numeric(performance(roc, "auc")@y.values)
auc #0.6956966

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
cat(sprintf("Accuracy: %0.3f\n", accuracy)) #0.797
cat(sprintf("Sensitivity: %0.3f\n", sensitivity)) #0.806
cat(sprintf("Specificity: %0.3f\n", specificity)) #0.649
cat(sprintf("Precision: %0.3f\n", precision)) #0.972
cat(sprintf("Recall: %0.3f\n", recall)) #0.806

# Find optimal threshold using Youden's J statistic
roc <- prediction(knn_prob[,2], test_data_normalized$Y)
perf <- performance(roc, "tpr", "fpr")
coords <- cbind(unlist(perf@x.values), unlist(perf@y.values))
j_stat <- coords[,2] - coords[,1]
opt_idx <- which.max(j_stat)
opt_threshold <- cbind(coords[opt_idx,1], coords[opt_idx,2])

# Plot ROC curve and optimal coordinate
plot(perf, col="blue")
abline(a=0, b=1, lty=2, col="gray")
points(opt_threshold, col="red", p=19, cex=1.5)

threshold_value = unlist(perf@alpha.values)[opt_idx] #Get the optimal value 0.2352941

knn_pred <- ifelse(knn_prob[,2] > threshold_value, "Yes", "No")