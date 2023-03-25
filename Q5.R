cat("\014")  # clear console
rm(list = ls())  # clear environment
graphics.off()  # clear plots tab

# load required packages
library(foreign)  # for reading SPSS data
library(stats)    # for linear regression
library(MASS)     # for backward selection
library(lmtest)   # Load the lmtest package - for Breusch-Pagan test for heteroscedasticity
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret) #For spliting training and test data sets
library(pROC)
library(car)
# read SPSS data into R as a data frame
original_data <- read.spss("/Users/vinotharanannarasa/BUSN9049/Assignment 1/Classification_Dataset.sav", to.data.frame = TRUE)
data <- original_data

#Summarise the data
summary(original_data)

#Identify unique variables
trimmed_data <- unique(data) #I got rid of 6 observation here

#Identify NAs
sum(is.na(trimmed_data)) #No empty variables

#summary also showed that nbumps6, nbumps7 and nbumps89 are all zeros, there let's get rid of those
trimmed_data <- trimmed_data[, !names(trimmed_data) %in% c("nbumps6","nbumps7","nbumps89")]

#Assumptions check - The dependent variable follows the binomial distribution - Met as the dependent vairable is binary in nature. "Yes" or "No"
table(trimmed_data$class,useNA = "always")

#Summarise again
summary(trimmed_data)
str(trimmed_data)
###################################Convert nominal and ordinal data to  factors
trimmed_data$seismic <- factor(trimmed_data$seismic)
trimmed_data$seismoacoustic <- factor(trimmed_data$seismoacoustic)
trimmed_data$shift <- factor(trimmed_data$shift)
trimmed_data$ghazard <- factor(trimmed_data$ghazard)
trimmed_data$class <- ifelse(trimmed_data$class == "Yes", 1, 0)
trimmed_data$class <- factor(trimmed_data$class, levels = c(0, 1), labels = c("No", "Yes"))

str(trimmed_data)

#Deal with highly correlated numeric variables
numeric_data_df <- trimmed_data[, names(trimmed_data) %in% c("genergy","gpuls",
                                                             "gdenergy","gdpuls","energy","maxenergy",
                                                             "nbumps","nbumps2","nbumps3","nbumps4","nbumps5")]

cor_matrix <- cor(numeric_data_df) # Create a correlation matrix of your data
upper_tri <- upper.tri(cor_matrix) # Extract the upper triangle of the correlation matrix (to avoid duplicate correlations)
cor_index <- which(abs(cor_matrix) > 0.7 & upper.tri(cor_matrix), arr.ind = TRUE) # Find the index of the elements in the upper triangle that have a correlation coefficient of greater than 0.7
cor_vars <- unique(c(rownames(cor_matrix)[cor_index[,1]], colnames(cor_matrix)[cor_index[,2]])) # Extract the variable names that are correlated
cor_vars <- unique(cor_vars) # Remove duplicates

# Extract the correlated variable pairs
for (i in 1:nrow(cor_index)) {
  cat(colnames(cor_matrix)[cor_index[i, 1]], "is correlated with", colnames(cor_matrix)[cor_index[i, 2]], "\n")
}

#Based on the above run:
#1) genergy is correlated with gpuls -remove genergy
#2) gdenergy is correlated with gdpuls - remove gdpuls
#3) nbumps is correlated with nbumps2 - remove nbumps2
#4) nbumps is correlated with nbumps3 - remove nbumps3
#5) nbumps5 is correlated with energy - remove nbumps5
#6) nbumps5 is correlated with maxenergy
#7) energy is correlated with maxenergy - remove maxenergy

trimmed_data <- trimmed_data[, !names(trimmed_data) %in% c("genergy","gdpuls","nbumps2","nbumps3","nbumps5","maxenergy")]

#Deal with highly correlated numeric variables - Round 2
numeric_data_df <- trimmed_data[, names(trimmed_data) %in% c("gdenergy","gpuls","energy","nbumps","nbumps4")]

cor_matrix <- cor(numeric_data_df) # Create a correlation matrix of your data
upper_tri <- upper.tri(cor_matrix) # Extract the upper triangle of the correlation matrix (to avoid duplicate correlations)
cor_index <- which(abs(cor_matrix) > 0.7 & upper.tri(cor_matrix), arr.ind = TRUE) # Find the index of the elements in the upper triangle that have a correlation coefficient of greater than 0.7
cor_vars <- unique(c(rownames(cor_matrix)[cor_index[,1]], colnames(cor_matrix)[cor_index[,2]])) # Extract the variable names that are correlated
cor_vars <- unique(cor_vars) # No strong correlations

summary(trimmed_data)

#Deal with nominal data - I checked that all tables had a count of > 5
nominal_data <- trimmed_data[, names(trimmed_data) %in% c("shift", "seismoacoustic", "ghazard")]
table1 <- table(nominal_data$shift, nominal_data$seismoacoustic)
result1 <- chisq.test(table1)

table2 <- table(nominal_data$shift, nominal_data$ghazard)
result2 <- chisq.test(table2)

table3 <- table(nominal_data$shift, nominal_data$ghazard)
result3 <- chisq.test(table3)

# Print the test statistics and p-values
print(result1)
print(result2)
print(result3)

# Check which pairs of variables are most strongly associated
if (result1$p.value < 0.05) {
  cat("There is evidence of a significant association between nominal_var1 and nominal_var2.\n")
}

if (result2$p.value < 0.05) {
  cat("There is evidence of a significant association between nominal_var1 and nominal_var3.\n")
}

if (result3$p.value < 0.05) {
  cat("There is evidence of a significant association between nominal_var2 and nominal_var3.\n")
}

#There is a significant association between all three nominal variables, therefore I will remove 2 of the three.
#I will remove "seismoacoustic", "ghazard"

trimmed_data <- trimmed_data[, !names(trimmed_data) %in% c("seismoacoustic","ghazard")]


#Check the frequency of the different character variables with the outcome variable
table(trimmed_data$seismic,trimmed_data$class, useNA = "always")
table(trimmed_data$shift,trimmed_data$class, useNA = "always")



#I might have to come back to this. Check through the R book
original_full_model <- glm(trimmed_data$class ~ ., data = trimmed_data, family = binomial)
summary(original_full_model)

#Split data into training and test datasets
set.seed(123) # Set the seed for reproducibility

# Split the dataset into 70% training and 30% test
train_index <- createDataPartition(trimmed_data$class, p = 0.7, list = FALSE)
train_data <- trimmed_data[train_index, ]
test_data <- trimmed_data[-train_index, ]

#Check if there are enough outcome points for both the training and test sets
table(train_data$class, useNA = "ifany")
table(test_data$class, useNA = "ifany")

#Check the frequency of the different character variables with the outcome variable - Training data
table(train_data$seismic,train_data$class, useNA = "always")
table(train_data$shift,train_data$class, useNA = "always")

#Fit logistics regression model
train_data_full_model <- glm(train_data$class ~ ., data = train_data, family = binomial)
backwards_model <- step(train_data_full_model, direction = "backward")
summary(backwards_model)

my_model <- glm(formula = train_data$class ~ seismic + shift + gpuls + gdenergy + 
                  nbumps + nbumps4, family = binomial, data = train_data)

#Check multicollinearity assumption
car::vif(backwards_model)

#Testing for linearity of logit
#Box Tidwell method
train_data$log_gpuls <- log(train_data$gpuls)
train_data$log_nbumps <- log(train_data$nbumps)

# Interaction terms
train_data$gpuls_log_gpuls <- train_data$gpuls * train_data$log_gpuls
train_data$nbumps_log_nbumps <- train_data$nbumps * train_data$log_nbumps
#train_data$nbumps4_log_nbumps4 <- train_data$nbumps4 * train_data$log_nbumps4

# Fit the logistic regression model with interaction terms
logit_check_model <- glm(class ~ gpuls + log_gpuls + gpuls_log_gpuls +
               nbumps + log_nbumps + nbumps_log_nbumps,
             data = train_data, family = binomial(link = "logit"))

summary(logit_check_model)

#I couldn't do the boxTidwell method or interaction between predictor and predictor*log(predictor). The former due to error and latter due to gdenergy having -ves
train_data$predicted_prob <- predict(my_model, type = "response")
train_data$deviance_residuals <- residuals(my_model, type = "deviance")

# Plot Lowess curve for gdenergy
ggplot(train_data, aes(x = gdenergy, y = deviance_residuals)) +
  geom_point() +
  stat_smooth(method = "loess", se = FALSE, linetype = "dashed", color = "red") +
  theme_minimal() +
  xlab("gdenergy") +
  ylab("Deviance Residuals")

# Plot smooth curve for nbumps4 using the gam method
ggplot(train_data, aes(x = nbumps4, y = deviance_residuals)) +
  geom_point() +
  stat_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k = 3), se = FALSE, linetype = "dashed", color = "red") +
  theme_minimal() +
  xlab("nbumps4") +
  ylab("Deviance Residuals")


#Since the results show that there is non-linearity - transformation may need to be applied

ggplot(train_data, aes(x = train_data$gpuls, y = predict(my_model, train_data, type = "response"))) + 
  geom_point() + 
  geom_smooth(method = "lm")

ggplot(train_data, aes(x = train_data$gdenergy, y = predict(my_model, train_data, type = "response"))) + 
  geom_point() + 
  geom_smooth(method = "lm")

ggplot(train_data, aes(x = train_data$nbumps, y = predict(my_model, train_data, type = "response"))) + 
  geom_point() + 
  geom_smooth(method = "lm")


ggplot(train_data, aes(x = train_data$nbumps4, y = predict(my_model, train_data, type = "response"))) + 
  geom_point() + 
  geom_smooth(method = "lm")


#Measure model performance
new_data <- test_data[, !names(test_data) %in% c("class")]

predicted_probs <- predict(my_model, newdata = new_data, type = "response")
summary(predicted_probs)

roc_curve <- roc(test_data$class, predicted_probs)
plot(roc_curve, print.thres = "best")
auc_score <- auc(roc_curve)
coords <- coords(roc_curve, "best")
optimal_threshold <- coords$threshold

actual <- test_data$class # replace "target_variable" with the name of your binary outcome variable
predicted <- factor(ifelse(predicted_probs > optimal_threshold, "Yes", "No")) # threshold predicted probabilities at 0.5
conf_matrix <- confusionMatrix(predicted, actual)
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
f1_score <- conf_matrix$byClass["F1"]

#Export data
# create a summary table of the model coefficients
summary_table <- data.frame(coef(summary(my_model)))

# add the odds ratios and 95% confidence intervals to the summary table
summary_table$odds_ratio <- exp(summary_table[, "Estimate"])
summary_table$ci_low <- exp(summary_table[, "Estimate"] - 1.96 * summary_table[, "Std..Error"])
summary_table$ci_high <- exp(summary_table[, "Estimate"] + 1.96 * summary_table[, "Std..Error"])

# reorder the columns of the summary table
summary_table <- summary_table[, c(1, 2, 6, 5, 7)]

# rename the columns of the summary table
colnames(summary_table) <- c("Variable", "Coeff.", "SE", "CI (lower)", "Odds Ratio", "CI (higher)")

# save the summary table as a CSV file
write.csv(summary_table, file = "logistic_regression_summary.csv", row.names = FALSE)
