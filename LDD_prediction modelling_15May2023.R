#Purpose:Code to develop ML models for LDD
#Developed by: Billy Ogwel
#Date:12 Jan 2023

library(randomForest)
library(caret)
library(e1071)
library(dplyr)
library(haven)
library(MLmetrics)
library(xgboost)
library(mice)
library(class)
library(Amelia)
library(neuralnet)
#library(fastAdaboost)
library(adabag)
library(Boruta)
library(ROSE)
#library(DMwR)
library(caretEnsemble)
library(pROC)
library(modelplotr)
library(SuperLearner)
#library(h2o)
library(caTools)
library(epiR)
library(rminer)
library(iml)
library(DALEX)
library(ResourceSelection) #computing hosmer-lemeshow values
library(rms) #Asessing calibration:Brier score and Spiegelhalter
library(VIM)
library(gridExtra)
library(hydroGOF)
library(remotes)
library(SMOTEWB)
library(githubinstall)
library(shapper)
library(boot)
library(ggplotify)
library(PRROC)
library(precrec) #for computing ROC & PR AUC & CI
library(epiDisplay) #tab1 function
library(gmodels) #cross table function
library(readxl)
library(synthpop)
library(scales)
#detach(package:semantic.dashboard, unload = TRUE)

remove.packages("Matrix")

install.packages("Matrix")
#shapper::install_shap()

#install_github("cran/prob-pr_auc")

#importing dataset
LDD_data <- read_dta("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/VIDA_diarr_dur_anal_12jan2023.dta")

#transforming target variable
LDD_data1 <- LDD_data %>%  mutate(ldd_bin=factor(ldd_bin,levels=c(1, 0), labels=c("Yes", "NO"))) 

#visualize the missing 
missmap(LDD_data1)
md.pattern(LDD_data1)

#Plotting patterns in missing data using VIM
mice_plot <- aggr(LDD_data1, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(LDD_data1), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))


#Use mice package to impute missing values
LDD_miss <- LDD_data1 %>%
  select(rotavax1,F4A_DRH_BELLYPAIN1,F4A_CUR_THIRSTY1,F4B_OUTCOME_PNEU,
         F4A_DRH_STRAIN1,fuel_clean,caretaker_edu,FLOOR_NAT_BIN) 

mice_mod <- mice(LDD_miss, method='rf', drop = FALSE, rfPackage = "randomForest")
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
LDD_data1$rotavax1 <- mice_complete$rotavax1
LDD_data1$F4A_DRH_BELLYPAIN1 <- mice_complete$F4A_DRH_BELLYPAIN1
LDD_data1$F4A_CUR_THIRSTY1 <- mice_complete$F4A_CUR_THIRSTY1
LDD_data1$F4B_OUTCOME_PNEU <- mice_complete$F4B_OUTCOME_PNEU
LDD_data1$F4A_DRH_STRAIN1 <- mice_complete$F4A_DRH_STRAIN1
LDD_data1$fuel_clean <- mice_complete$fuel_clean
LDD_data1$caretaker_edu <- mice_complete$caretaker_edu
LDD_data1$FLOOR_NAT_BIN <- mice_complete$FLOOR_NAT_BIN

missmap(LDD_data1)

save(LDD_data1, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_data1.Rda") 

#write_dta(hucs_data1, "C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Careseeking/Data/hucs_data1_imputed.dta")

load("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_data1.Rda")


#Running feature selection for LDD-including pre-enrolment days
LDD_data2b <- LDD_data1 %>% dplyr::select(-CHILDID,-enroll_date) #F4B_OUTCOME_MLNT

LDD_data2b_ <- LDD_data2b %>%
  rename(Diarr_days=F4A_DRH_DAYS,Vesikari_score=VESIKARI_SCORE, Vomit_days=F4A_DAYS_VOMIT2,
         Vomit=F4A_ANY_VOMIT, freq_vomit=F4A_FREQ_VOMIT1, rect_strain=F4A_DRH_STRAIN1, 
         Stool_count=F4A_DAILY_MAX1, Belly_pain=F4A_DRH_BELLYPAIN1, Cough=F4A_DRH_COUGH1,
         Pneumonia=F4B_OUTCOME_PNEU, Fast_breath=F4A_CUR_FASTBREATH1, Under_nutr=UNDER_NUTR, 
         Thirsty=F4A_CUR_THIRSTY1, Natural_floor=FLOOR_NAT_BIN, Home_ORS=F4A_HOMETRT_ORS,
         Agegroup=AGEGROUP, Skin_turgor=F4B_SKIN1, Dry_mouth=F4B_MOUTH, Mental_status=F4B_MENTAL,
         Rotavirus_vacc=rotavax1)

# Perform Boruta search
boruta_output1 <- Boruta(ldd_bin ~ ., data=na.omit(LDD_data2b_), doTrace=1)
# Get significant variables including tentatives
boruta_signif1 <- getSelectedAttributes(boruta_output1, withTentative = TRUE)
print(boruta_signif1)

# Variable Importance Scores
imps <- attStats(boruta_output1)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
par(mar = c(8,5,2,2))
LDD_FS2 <- plot(boruta_output1, cex.axis=.9, las=2, xlab="", main="Longer Diarrheal Duration Feature Selection")

#Droping rejected features-keeping selected and tentative
LDD_data3b <- LDD_data2b_ %>% dplyr::select(ldd_bin,Diarr_days,Vesikari_score,Agegroup,Vomit_days,
                                    resp_rate,freq_vomit,Vomit,Rotavirus_vacc,Skin_turgor,
                                    Stool_count) #breast_feed

# LDD_data3c <- LDD_data2b %>% dplyr::select(ldd_bin,VESIKARI_SCORE,AGEGROUP,F4A_DAYS_VOMIT2,
#                                            resp_rate,F4A_FREQ_VOMIT1,F4A_ANY_VOMIT,rotavax1,F4B_SKIN1,
#                                            F4A_DAILY_MAX1) #F4A_DRH_DAYS

#Creating training set and test set for the two scenarios
# set.seed(7777777)
# test_index_ldd <- createDataPartition(LDD_data3$ldd_bin, times = 1, p = 0.75, list = FALSE)
# train_set_ldd  <- LDD_data3[test_index_ldd, ]
# test_set_ldd <- LDD_data3[-test_index_ldd, ]
# 
# set.seed(1111111)
# test_index_ldd_ <- createDataPartition(LDD_data3a$ldd_bin, times = 1, p = 0.75, list = FALSE)
# train_set_ldd_  <- LDD_data3a[test_index_ldd_, ]
# test_set_ldd_ <- LDD_data3a[-test_index_ldd_, ]

set.seed(1111111)
test_index_ldd_1 <- createDataPartition(LDD_data3b$ldd_bin, times = 1, p = 0.75, list = FALSE)
train_set_ldd_1  <- LDD_data3b[test_index_ldd_1, ]
test_set_ldd_1 <- LDD_data3b[-test_index_ldd_1, ]

X<- train_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y<- train_set_ldd_1 %>% dplyr::select(ldd_bin)  %>% mutate(ldd_bin=as.numeric(ldd_bin))
X1 <- test_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y1 <- test_set_ldd_1 %>% dplyr::select(ldd_bin) %>% mutate(ldd_bin=as.numeric(ldd_bin))


# Define the control
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred  = data$pred, y_true = data$obs, positive = lev[1])
  c(F1 = f1_val)
}

trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid",
                          summaryFunction=f1,
                          classProbs = TRUE)
trControl1 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=f1,
                           classProbs = TRUE,
                           sampling = "up")

trControl2 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=f1,
                           classProbs = TRUE,
                           sampling = "down")

trControl3 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=f1,
                           classProbs = TRUE,
                           sampling = "rose")


##---------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------##
#1- Predicting LDD including pre-enrolment days and dropping rectal straining

set.seed(7777777)
rf_default_Ma <- train(ldd_bin ~ ., method = "rf", data=train_set_ldd_1, metric= "F1",
                        trControl = trControl,threshold = 0.3)

rf_prediction_Ma <-predict(rf_default_Ma, test_set_ldd_1,type = "raw")
confusionMatrix(rf_prediction_Ma, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
rf_def_data <- as.table(matrix(c(57,38,62,213), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_def_rval <- epi.tests(rf_def_data, conf.level = 0.95,digits = 3)
print(rf_def_rval)

#f1 score 95% ci
f1_rf_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(rf_prediction_Ma) %>%
  rename(obs=ldd_bin, pred=rf_prediction_Ma) %>% mutate(obs=as.numeric(obs),
                                                        pred=as.numeric(pred))

bootout<-boot(data=f1_rf_data,
              R=5000,
              statistic=f1)

boot.ci(bootout,type="norm")
f1(f1_rf_data)

roc_rf_def <- roc(test_set_ldd_1$ldd_bin,
              predict(rf_default_Ma, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))

pROC::auc(roc_rf_def)
ci.auc(roc_rf_def)

#Calculating AUPRC
rf_prediction_Ma <-predict(rf_default_Ma, test_set_ldd_1,type ="prob")[,2]

rf_s1a<- rf_prediction_Ma[1:74]
rf_s2a <- rf_prediction_Ma[75:148]
rf_s3a <- rf_prediction_Ma[149:222]
rf_s4a <- rf_prediction_Ma[223:296]
rf_s5a <- rf_prediction_Ma[297:370]


rf_l1 <- test_set_ldd_1$ldd_bin[1:74]
rf_l2 <- test_set_ldd_1$ldd_bin[75:148]
rf_l3 <- test_set_ldd_1$ldd_bin[149:222]
rf_l4 <- test_set_ldd_1$ldd_bin[223:296]
rf_l5 <- test_set_ldd_1$ldd_bin[297:370]

rf_s_a<- join_scores(rf_s1a,rf_s2a,rf_s3a,rf_s4a,rf_s5a)
rf_l <- join_labels(rf_l1,rf_l2,rf_l3,rf_l4,rf_l5)

rfmdat_ <- mmdata(scores=rf_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

rf_curves_ <- evalmod(scores=rf_prediction_Ma, labels= test_set_ldd_1$ldd_bin)
rf_curves1_ <- evalmod(rfmdat_)

auc(rf_curves_)
auc(rf_curves1_)
auc_ci(rf_curves1_,alpha = 0.05, dtype = "normal")


#Running Over-sampling

set.seed(1111111)
rf_default_M1a <- train(ldd_bin ~ ., method = "rf", data=train_set_ldd_1, metric= "F1",
                        trControl = trControl1,threshold = 0.3)

rf_prediction_M1a <-predict(rf_default_M1a, test_set_ldd_1,type = "raw")
confusionMatrix(rf_prediction_M1a, test_set_ldd_1$ldd_bin)

varImp(rf_default_M1a)

#*######################Parameter Tuning for Random Forest############
#selecting best mtry
set.seed(1111111)
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(ldd_bin ~ .
                 , method = "rf", data=train_set_ldd_1, metric= "F1",
                 tuneGrid = tuneGrid, trControl = trControl1)
print(rf_mtry)
best_mtry <- rf_mtry$bestTune$mtry
best_mtry
#accuracy highest at mtry=1

#selecting best maxnodes

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(15: 30)) {
  set.seed(1111111)
  rf_maxnode <- train(ldd_bin ~ .,
                      data = train_set_ldd_1,
                      method = "rf",
                      metric = "F1",
                      tuneGrid = tuneGrid,
                      trControl = trControl1,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#best maxnode n=22

#Search the best ntrees

store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(1111111)
  rf_maxtrees <- train(ldd_bin ~ .,
                       data = train_set_ldd_1,
                       method = "rf",
                       metric = "F1",
                       tuneGrid = tuneGrid,
                       trControl = trControl1,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 30,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)
#ntree=300 best ntree

#training final model
#mtry=1
#maxnodes n=22
#ntree=1000

set.seed(1111111)
fit_rf <- train(ldd_bin ~ .,
                train_set_ldd_1,
                method = "rf",
                metric = "F1",
                tuneGrid = tuneGrid,
                trControl = trControl1,
                importance = TRUE,
                nodesize = 14,
                ntree = 1000,
                maxnodes = 22)

print(fit_rf)

#model validation
rf_pred_fit <-predict(fit_rf, test_set_ldd_1,type = "raw")
confusionMatrix(rf_pred_fit, test_set_ldd_1$ldd_bin)

#optimized model doesnt perform better than the default model

rf_prediction_M1a <-predict(rf_default_M1a, test_set_ldd_1,type = "raw")
confusionMatrix(rf_prediction_M1a, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_RF <- matrix(c(96, 65, 23, 186), nrow = 2, byrow = TRUE)
# #saving model
saveRDS(rf_default_M1a, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/rf_LDD.rds")

#loading model 
rf_default_M1a= readRDS("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/rf_LDD.rds")



#computing 95% CI for performance metrics
rf_data <- as.table(matrix(c(96,65,23,186), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval <- epi.tests(rf_data, conf.level = 0.95,digits = 3)
print(rf_rval)

#computing 95% CI for F1-score
f1_rf_data1 <- test_set_ldd_1 %>%
 dplyr:: select(ldd_bin) %>% cbind(rf_prediction_M1a) %>%
  rename(obs=ldd_bin, pred=rf_prediction_M1a) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="basic")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(test_set_ldd_1$ldd_bin,
              predict(rf_default_M1a, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))


roc1<-ggroc(roc_rf, colour="Red") +
  ggtitle(paste0("Random Forest ROC Curve","(AUC=", round(pROC::auc(roc_rf),digits=4), ")")) +
  theme_minimal()

roc1

## Now plot
roc1a <- as.ggplot(function() plot(roc_rf, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc1a <- roc1a +
  labs(title=paste0("Random Forest ROC Curve","(AUC=", round(pROC::auc(roc_rf),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc1a

pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_prediction_M1a <-predict(rf_default_M1a, test_set_ldd_1,type ="prob")[,2]

rf_s1<- rf_prediction_M1a[1:74]
rf_s2 <- rf_prediction_M1a[75:148]
rf_s3 <- rf_prediction_M1a[149:222]
rf_s4 <- rf_prediction_M1a[223:296]
rf_s5 <- rf_prediction_M1a[297:370]


rf_l1 <- test_set_ldd_1$ldd_bin[1:74]
rf_l2 <- test_set_ldd_1$ldd_bin[75:148]
rf_l3 <- test_set_ldd_1$ldd_bin[149:222]
rf_l4 <- test_set_ldd_1$ldd_bin[223:296]
rf_l5 <- test_set_ldd_1$ldd_bin[297:370]

rf_s<- join_scores(rf_s1,rf_s2,rf_s3,rf_s4,rf_s5)
rf_l <- join_labels(rf_l1,rf_l2,rf_l3,rf_l4,rf_l5)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

rf_curves <- evalmod(scores=rf_prediction_M1a, labels= test_set_ldd_1$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_rf <- explain( model = rf_default_M1a, data = X, y = Y,label = "Random forest",type = "classification")
saveRDS(explainer_rf, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/explainer_rf.rds")


explainer_rf_1 <- update_data(explainer_rf, data = X1, y = Y1)


resids_rf <- model_performance(explainer_rf_1)
plot(resids_rf)

mp_rf <- model_parts(explainer_rf_1, type = "difference")
plot (mp_rf, show_boxplots = FALSE)
#axis(2,labels=format(mp_rf,scientific=FALSE))



rf_shap <-predict_parts(explainer = explainer_rf,
                        new_observation = X1,
                        type = "shap",
                        B = 10 #number of reorderings - start small
)

rf_shap1 <-  plot(rf_shap, show_boxplots = FALSE)
rf_shap1


#Assessing model calibration
rf_prediction_M1a_ <-predict(rf_default_M1a, test_set_ldd_1,type = "prob")

test_set_ldd_1_ <- test_set_ldd_1 %>% 
  mutate(ldd_bin=as.numeric(test_set_ldd_1$ldd_bin)) %>%
  mutate(ldd_bin=if_else(ldd_bin==2,0,ldd_bin))


val.prob(rf_prediction_M1a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))

#Plotting calibration plot:

# Generate predicted probabilities for the positive class
rf_probs_M1a <- predict(rf_default_M1a, test_set_ldd_1, type = "prob")[, 2]

# Create a dataframe with true outcomes and predicted probabilities
calibration_data <- data.frame(
  actual = as.numeric(test_set_ldd_1$ldd_bin),
  predicted = rf_probs_M1a
)

# Divide predicted probabilities into bins (e.g., deciles)
calibration_data <- calibration_data %>%
  mutate(bin = cut(predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE),
         actual=if_else(actual==2,1,0)) %>%
  group_by(bin) %>%
  summarize(
    bin_mean_pred = mean(predicted),
    bin_mean_obs = mean(actual),
    count = n()
  ) %>%
  na.omit()  # Remove any NA rows

# Calibration plot
perfect_calibration <- data.frame(x = c(0, 1), y = c(0, 1))

calibration_plot <- ggplot(calibration_data, aes(x = bin_mean_pred, y = bin_mean_obs)) +
  geom_smooth(
    aes(color = "Calibration Curve"), 
    method = "loess",  # Use LOESS smoothing method
    size = 1,          # Line thickness
    se = TRUE,         # Display confidence interval as ribbon
    alpha = 0.2        # Set transparency for the ribbon
  ) +  # Smoothed calibration curve with ribbon
  geom_point(aes(color = "Calibration Curve"), size = 2) +  # Points
  geom_line(
    data = perfect_calibration, aes(x = x, y = y, linetype = "Perfect Calibration"),
    color = "red", size = 1
  ) +  # Perfect calibration line
  scale_color_manual(
    name = " ",  # Legend title
    values = c("Calibration Curve" = "black"),  # Colors for calibration curve
    labels = c("Calibration Curve")  # Legend labels for the curve
  ) +
  scale_linetype_manual(
    name = " ",  # Legend title
    values = c("Perfect Calibration" = "dashed"),  # Linetype for the perfect line
    labels = c("Perfect Calibration")  # Legend labels for the perfect line
  ) +
  labs(
    title = "Calibration Plot for Champion Model: Random Forest",
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.position = "top",  # Move legend to the top
    legend.direction = "horizontal"
  ) +
  scale_y_continuous(breaks = seq(0, 1, 0.1), labels = scales::percent_format()) +
  scale_x_continuous(breaks = seq(0, 1, 0.1), labels = scales::percent_format())

ggsave("C:/Users/Bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Tableshells/calibration_plot.tiff", plot = calibration_plot, width = 22, height = 14, dpi = 300)

# Computing business value of model using modelplotr

scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("train_set_ldd_1","test_set_ldd_1"),
                                               dataset_labels = list("train data","test data"),
                                               models = list("rf_default_M1a"),  
                                               model_labels = list("Random Forest"), 
                                               target_column="ldd_bin",
                                               ntiles = 100)

plot_input <- plotting_scope(prepared_input = scores_and_ntiles)

#cummulative gains 
plot_cumgains(data = plot_input)

#Cumulative lift
plot_cumlift(data = plot_input)

#Response plot
plot_response(data = plot_input)

#Cumulative response plot
plot_cumresponse(data = plot_input)

#multiple plots
plot_multiplot(data = plot_input)

#Temporal Validation
#importing dataset
LDD_data_efgh <- read_dta("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/EFGH_dur_anal.dta")

#transforming target variable
LDD_data_efgh1 <- LDD_data_efgh %>%  mutate(ldd_bin=factor(ldd_bin,levels=c(1, 0), labels=c("Yes", "NO"))) 

missmap(LDD_data_efgh1)

mal_miss <- LDD_data_efgh1 %>%
  dplyr::select(Rotavirus_vacc,Stool_count)

mice_mod <- mice(mal_miss, method='rf', drop = FALSE, rfPackage = "randomForest")
mice_complete <- complete(mice_mod)


#Transfer the predicted missing values into the main data set
LDD_data_efgh1$Rotavirus_vacc <- mice_complete$Rotavirus_vacc

save(LDD_data_efgh1, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_data_efgh1.Rda") 

LDD_data_efgh1a <- LDD_data_efgh1 %>%
  dplyr::select(-MSD)

rf_prediction_M1a <-predict(rf_default_M1a, LDD_data_efgh1a,type = "raw")
confusionMatrix(rf_prediction_M1a, LDD_data_efgh1a$ldd_bin, mode='everything')

rf_data_def1 <- as.table(matrix(c(26,86,43,530), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval_def1 <- epi.tests(rf_data_def1, conf.level = 0.95,digits = 3)
print(rf_rval_def1)


#computing 95% CI for F1-score
f1_rf_data1 <- LDD_data_efgh1a %>%
  dplyr:: select(ldd_bin) %>% cbind(rf_prediction_M1a) %>%
  rename(obs=ldd_bin, pred=rf_prediction_M1a) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="basic")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(LDD_data_efgh1a$ldd_bin,
              predict(rf_default_M1a, LDD_data_efgh1a, type = "prob")[,1],
              levels = rev(levels(LDD_data_efgh1a$ldd_bin)))


pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_prediction_M1a <-predict(rf_default_M1a, LDD_data_efgh1a,type ="prob")[,2]

rf_s1<- rf_prediction_M1a[1:341]
rf_s2 <- rf_prediction_M1a[342:682]



rf_l1 <- LDD_data_efgh1a$ldd_bin[1:341]
rf_l2 <- LDD_data_efgh1a$ldd_bin[342:682]


rf_s<- join_scores(rf_s1,rf_s2)
rf_l <- join_labels(rf_l1,rf_l2)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:2)

rf_curves <- evalmod(scores=rf_prediction_M1a, labels= LDD_data_efgh1a$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")

#Plotting temporal validation for champion Model 

LDD_val <- read_excel("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Tableshells/LDD_tables_15May2023.xlsx",sheet='Sheet2', col_names = TRUE)


LDD_val1 <- LDD_val %>%
    mutate(Study=factor(Study,levels=c("Internal_validation","Temporal_validation")),
           metric=factor(metric,levels=c("Sensitivity", "Specificity", "PPV","NPV","F1","AUC","PRAUC")),
           estimate=as.numeric(estimate))

tv1 <- ggplot(LDD_val1, aes(fill=Study, y=estimate, x=metric)) + 
  geom_bar(position="dodge", stat="identity")+
  #geom_errorbar(aes(ymin=lower_limit, ymax=upper_limit)) +
  scale_y_continuous(breaks=seq(0, 100, 10)) +
  #geom_text(aes(label = estimate), vjust = -0.5) +
  geom_text(aes(label=estimate), position=position_dodge(width=0.9), vjust=-0.25) +
  ylab("Performance (%)") +
  xlab("Metrics") +
  labs(title="EFGH MAD Cases") + #"VIDA_EFGH Data"
  theme_bw()
  

tv1

#Using MSD cases in EFGH
LDD_data_efgh1b <- LDD_data_efgh1 %>% filter(MSD==1) %>%
  dplyr::select(-MSD)

rf_prediction_M1a <-predict(rf_default_M1a, LDD_data_efgh1b,type = "raw")
confusionMatrix(rf_prediction_M1a, LDD_data_efgh1b$ldd_bin, mode='everything')

rf_data_def1 <- as.table(matrix(c(19,55,21,227), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval_def1 <- epi.tests(rf_data_def1, conf.level = 0.95,digits = 3)
print(rf_rval_def1)


#computing 95% CI for F1-score
f1_rf_data1 <- LDD_data_efgh1b %>%
  dplyr:: select(ldd_bin) %>% cbind(rf_prediction_M1a) %>%
  rename(obs=ldd_bin, pred=rf_prediction_M1a) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="norm")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(LDD_data_efgh1b$ldd_bin,
              predict(rf_default_M1a, LDD_data_efgh1b, type = "prob")[,1],
              levels = rev(levels(LDD_data_efgh1b$ldd_bin)))


pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_prediction_M1a <-predict(rf_default_M1a, LDD_data_efgh1b,type ="prob")[,2]

rf_s1<- rf_prediction_M1a[1:161]
rf_s2 <- rf_prediction_M1a[162:322]



rf_l1 <- LDD_data_efgh1b$ldd_bin[1:161]
rf_l2 <- LDD_data_efgh1b$ldd_bin[162:322]


rf_s<- join_scores(rf_s1,rf_s2)
rf_l <- join_labels(rf_l1,rf_l2)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:2)

rf_curves <- evalmod(scores=rf_prediction_M1a, labels= LDD_data_efgh1b$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")

#Plotting temporal validation for champion Model 

LDD_val_ <- read_excel("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Tableshells/LDD_tables_15May2023.xlsx",sheet='Sheet2a', col_names = TRUE)


LDD_val1_ <- LDD_val_ %>%
  mutate(Study=factor(Study,levels=c("Internal_validation","Temporal_validation")),
         metric=factor(metric,levels=c("Sensitivity", "Specificity", "PPV","NPV","F1","AUC","PRAUC")),
         estimate=as.numeric(estimate))

tv1a <- ggplot(LDD_val1_, aes(fill=Study, y=estimate, x=metric)) + 
  geom_bar(position="dodge", stat="identity")+
  #geom_errorbar(aes(ymin=lower_limit, ymax=upper_limit)) +
  scale_y_continuous(breaks=seq(0, 100, 10)) +
  #geom_text(aes(label = estimate), vjust = -0.5) +
  geom_text(aes(label=estimate), position=position_dodge(width=0.9), vjust=-0.25) +
  ylab("Performance (%)") +
  xlab("Metrics") +
  labs(title="EFGH MSD Cases") +
  theme_bw()

tv1a

grid.arrange(tv1,tv1a, ncol = 1 )  
# #Generating Explainer
# X2 <- LDD_data_efgh1 %>% dplyr::select(-ldd_bin)
# Y2 <- LDD_data_efgh1 %>% dplyr::select(ldd_bin) %>% mutate(ldd_bin=as.numeric(ldd_bin))
# 
# #computing shapley values using DALEX
# explainer_rf <- explain( model = rf_default_M1a, data = X, y = Y,label = "Random forest",type = "classification")
# 
# 
# rf_shap2 <-predict_parts(explainer = explainer_rf,
#                         new_observation = X2,
#                         type = "shap",
#                         B = 10 #number of reorderings - start small
# )
# 
# rf_shap3 <-  plot(rf_shap2, show_boxplots = FALSE)
# rf_shap3


#----------------------------------------------------------------------------#
#Generating Synthetic data
LDD_data_efgh2 <- LDD_data_efgh1 %>% mutate(Vomit=as.factor(Vomit), 
                                            Vesikari_score=as.factor(Vesikari_score),
                                            Agegroup=as.factor(Agegroup),freq_vomit=as.factor(freq_vomit),
                                            Rotavirus_vacc=as.factor(Rotavirus_vacc),Stool_count=as.factor(Stool_count),
                                            Skin_turgor=as.factor(Skin_turgor))

EFGH_syn <- syn(LDD_data_efgh2, m = 1,k=2964, method= "cart", 
                cart.minbucket = 10, seed = 7777777)
EFGH_syn1 <- EFGH_syn$syn 
missmap(EFGH_syn1)

mal_miss <- EFGH_syn1 %>%
  dplyr::select(Rotavirus_vacc,Stool_count)

mice_mod <- mice(mal_miss, method='rf', drop = FALSE, rfPackage = "randomForest")
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
EFGH_syn1$Rotavirus_vacc <- mice_complete$Rotavirus_vacc


EFGH_syn2 <- EFGH_syn1 %>% mutate(Vomit=as.numeric(Vomit), 
                                  Vesikari_score=as.numeric(Vesikari_score),Agegroup=as.numeric(Agegroup),
                                  freq_vomit=as.numeric(freq_vomit),Rotavirus_vacc=as.numeric(Rotavirus_vacc),
                                  Stool_count=as.numeric(Stool_count),Skin_turgor=as.numeric(Skin_turgor))

save(EFGH_syn2, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/EFGH_syn2.Rda") 

LDD_new <- rbind(LDD_data3b, EFGH_syn2)

missmap(LDD_new)

set.seed(1111111)
test_index_ldd_1 <- createDataPartition(LDD_new$ldd_bin, times = 1, p = 0.75, list = FALSE)
train_set_ldd_1  <- LDD_new[test_index_ldd_1, ]
test_set_ldd_1 <- LDD_new[-test_index_ldd_1, ]

X<- train_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y<- train_set_ldd_1 %>% dplyr::select(ldd_bin)  %>% mutate(ldd_bin=as.numeric(ldd_bin))
X1 <- test_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y1 <- test_set_ldd_1 %>% dplyr::select(ldd_bin) %>% mutate(ldd_bin=as.numeric(ldd_bin))


#Running Over-sampling

set.seed(7777777)
rf_syn <- train(ldd_bin ~ ., method = "rf", data=train_set_ldd_1, metric= "F1",
                        trControl = trControl1,threshold = 0.3)

rf_pred_syn <-predict(rf_syn, test_set_ldd_1,type = "raw")
confusionMatrix(rf_pred_syn, test_set_ldd_1$ldd_bin)

rf_data_def1 <- as.table(matrix(c(131,151,66,763), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval_def1 <- epi.tests(rf_data_def1, conf.level = 0.95,digits = 3)
print(rf_rval_def1)


f1_rf_data1 <- test_set_ldd_1 %>%
  dplyr:: select(ldd_bin) %>% cbind(rf_pred_syn) %>%
  rename(obs=ldd_bin, pred=rf_pred_syn) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="basic")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(test_set_ldd_1$ldd_bin,
              predict(rf_syn, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))


pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_pred_syn1a <-predict(rf_syn, test_set_ldd_1,type ="prob")[,2]

rf_s1<- rf_pred_syn1a[1:341]
rf_s2 <- rf_pred_syn1a[342:682]


rf_l1 <- LDD_data_efgh1$ldd_bin[1:341]
rf_l2 <- LDD_data_efgh1$ldd_bin[342:682]


rf_s<- join_scores(rf_s1,rf_s2)
rf_l <- join_labels(rf_l1,rf_l2)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:2)

rf_curves <- evalmod(scores=rf_pred_syn1a, labels= test_set_ldd_1$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")

#Temporal validation of model developed using synthetic data

rf_pred_syn1a <-predict(rf_syn, LDD_data_efgh1,type = "raw")
confusionMatrix(rf_pred_syn1a, LDD_data_efgh1$ldd_bin, mode='everything')

rf_data_def1 <- as.table(matrix(c(38,161,31,452), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval_def1 <- epi.tests(rf_data_def1, conf.level = 0.95,digits = 3)
print(rf_rval_def1)

#computing 95% CI for F1-score
f1_rf_data1 <- LDD_data_efgh1 %>%
  dplyr:: select(ldd_bin) %>% cbind(rf_pred_syn1a) %>%
  rename(obs=ldd_bin, pred=rf_pred_syn1a) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="basic")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(LDD_data_efgh1$ldd_bin,
              predict(rf_syn, LDD_data_efgh1, type = "prob")[,1],
              levels = rev(levels(LDD_data_efgh1$ldd_bin)))



pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_prediction_M1a <-predict(rf_syn, LDD_data_efgh1,type ="prob")[,2]

rf_s1<- rf_prediction_M1a[1:341]
rf_s2 <- rf_prediction_M1a[342:682]

rf_l1 <- LDD_data_efgh1$ldd_bin[1:341]
rf_l2 <- LDD_data_efgh1$ldd_bin[342:682]


rf_s<- join_scores(rf_s1,rf_s2)
rf_l <- join_labels(rf_l1,rf_l2)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:2)

rf_curves <- evalmod(scores=rf_prediction_M1a, labels= LDD_data_efgh1$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")


#Plotting temporal validation for champion Model using synthetic data

LDD_val2 <- read_excel("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Tableshells/LDD_tables_15May2023.xlsx",sheet='Sheet3', col_names = TRUE)


LDD_val3 <- LDD_val2 %>%
  mutate(Study=factor(Study,levels=c("Internal_validation","Temporal_validation")),
         metric=factor(metric,levels=c("Sensitivity", "Specificity", "PPV","NPV","F1","AUC","PRAUC")),
         estimate=as.numeric(estimate))

tv2 <- ggplot(LDD_val3, aes(fill=Study, y=estimate, x=metric)) + 
  geom_bar(position="dodge", stat="identity")+
  #geom_errorbar(aes(ymin=lower_limit, ymax=upper_limit)) +
  scale_y_continuous(breaks=seq(0, 100, 10)) +
  #geom_text(aes(label = estimate), vjust = -0.5) +
  geom_text(aes(label=estimate), position=position_dodge(width=0.9), vjust=-0.25) +
  ylab("Performance (%)") +
  xlab("Metrics") +
  labs(title="VIDA_Synthetic_EFGH Data") +
  theme_bw()


tv2

grid.arrange(tv1,tv2, ncol = 1 )


#___________________________________________________________________________________________________#
#2.Gradient Boosting Algorithm

set.seed(7777777)
gbm_ldd_model <- train( ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "xgbTree", 
                         trControl = trControl,metric= "F1") #shrinkage = 0.01,

gbm_prediction <-predict(gbm_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(gbm_prediction, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
gbm_data_def <- as.table(matrix(c(63,36,56,215), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
gbm_rval_def <- epi.tests(gbm_data_def, conf.level = 0.95,digits = 3)
print(gbm_rval_def)

#computing 95% CI for F1-score
f1_gbm_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(gbm_prediction) %>%
  rename(obs=ldd_bin, pred=gbm_prediction) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_gbm<-boot(data=f1_gbm_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_gbm,type="norm")
f1(f1_gbm_data1)

roc_gbm_def <- roc(test_set_ldd_1$ldd_bin,
               predict(gbm_ldd_model, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
pROC::auc(roc_gbm_def)
ci.auc(roc_gbm_def)


#Calculating AUPRC
gbm_prediction <-predict(gbm_ldd_model, test_set_ldd_1,type ="prob")[,2]


gbm_s1a<- gbm_prediction[1:74]
gbm_s2a <- gbm_prediction[75:148]
gbm_s3a <- gbm_prediction[149:222]
gbm_s4a <- gbm_prediction[223:296]
gbm_s5a <- gbm_prediction[297:370]

gbm_s_a<- join_scores(gbm_s1a,gbm_s2a,gbm_s3a,gbm_s4a,gbm_s5a)
gbm_mdat_ <- mmdata(scores=gbm_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

gbm_curves_ <- evalmod(scores=gbm_prediction, labels= test_set_ldd_1$ldd_bin)
gbm_curves1_ <- evalmod(gbm_mdat_)

auc(gbm_curves_)
auc(gbm_curves1_)
auc_ci(gbm_curves1_,alpha = 0.05, dtype = "normal")

#running over-sampling
set.seed(7777777)
gbm_ldd_model1 <- train( ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "xgbTree", 
                         trControl = trControl1,metric= "F1") #shrinkage = 0.01,

gbm_prediction1 <-predict(gbm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(gbm_prediction1, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_gbm <- matrix(c(86,59,33,192), nrow = 2, byrow = TRUE)


#computing 95% CI for performance metrics
gbm_data <- as.table(matrix(c(86,59,33,192), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
gbm_rval <- epi.tests(gbm_data, conf.level = 0.95,digits = 3)
print(gbm_rval)

#computing 95% CI for F1-score
f1_gbm_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(gbm_prediction1) %>%
  rename(obs=ldd_bin, pred=gbm_prediction1) %>% mutate(obs=as.numeric(obs),
                                                       pred=as.numeric(pred))

bootout_gbm1 <-boot(data=f1_gbm_data,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_gbm1,type="norm")
f1(f1_gbm_data)

#generating ROC Curve
roc_gbm <- roc(test_set_ldd_1$ldd_bin,
               predict(gbm_ldd_model1, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc2<-ggroc(roc_gbm, colour="Red") +
  ggtitle(paste0("Gradient Boosting ROC Curve","(AUC=", round(pROC::auc(roc_gbm),digits=4), ")")) +
  theme_minimal()

roc2

## Now plot
roc2a <- as.ggplot(function() plot(roc_gbm, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc2a <- roc2a +
  labs(title=paste0("Gradient Boosting ROC Curve","(AUC=", round(pROC::auc(roc_gbm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc2a

pROC::auc(roc_gbm)
ci.auc(roc_gbm)

#Calculating AUPRC
gbm_prediction1 <-predict(gbm_ldd_model1, test_set_ldd_1,type ="prob")[,2]


gbm_s1<- gbm_prediction1[1:74]
gbm_s2 <- gbm_prediction1[75:148]
gbm_s3 <- gbm_prediction1[149:222]
gbm_s4 <- gbm_prediction1[223:296]
gbm_s5 <- gbm_prediction1[297:370]

gbm_s<- join_scores(gbm_s1,gbm_s2,gbm_s3,gbm_s4,gbm_s5)
gbm_mdat <- mmdata(scores=gbm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

gbm_curves <- evalmod(scores=gbm_prediction1, labels= test_set_ldd_1$ldd_bin)
gbm_curves1 <- evalmod(gbm_mdat)

auc(gbm_curves)
auc(gbm_curves1)
auc_ci(gbm_curves1,alpha = 0.05, dtype = "normal")

#computing shapley values using DALEX
explainer_gbm <- explain( model = gbm_ldd_model1, data = X, y = Y,label = "Gradient Boosting",type = "classification")

explainer_gbm1 <- update_data(explainer_gbm, data = X1, y = Y1)


resids_gbm <- model_performance(explainer_gbm1)
plot(resids_gbm)

mp_gbm <- model_parts(explainer_gbm1, type = "difference")
plot (mp_gbm, show_boxplots = FALSE)


gbm_shap <-predict_parts(explainer = explainer_gbm,
                        new_observation = X1,
                        type = "shap",
                        B = 10 #number of reorderings - start small
)

gbm_shap1 <- plot(gbm_shap, show_boxplots = FALSE)
gbm_shap1

#Assessing Calibration

gbm_prediction1a <-predict(gbm_ldd_model1, test_set_ldd_1,type = "prob")

val.prob(gbm_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))

#----------------------------------------------------------------------------#
#3.Naive Bayes Algorithm -hyperparameter tuning not recommended for NB

set.seed(3333333)
nb_ldd_model <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "naive_bayes",
  trControl = trControl, metric= "F1")

nb_prediction <-predict(nb_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(nb_prediction, test_set_ldd_1$ldd_bin, mode="everything")


#computing 95% CI for performance metrics
nb_data_def <- as.table(matrix(c(59,32,60,219), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
nb_rval_def <- epi.tests(nb_data_def, conf.level = 0.95,digits = 3)
print(nb_rval_def)

#computing 95% CI for F1-score
f1_nb_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nb_prediction) %>%
  rename(obs=ldd_bin, pred=nb_prediction) %>% mutate(obs=as.numeric(obs),
                                                       pred=as.numeric(pred))

bootout_nb <-boot(data=f1_nb_data,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_nb,type="basic")
f1(f1_nb_data)

roc_nb_def <- roc(test_set_ldd_1$ldd_bin,
              predict(nb_ldd_model, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))
pROC::auc(roc_nb_def)
ci.auc(roc_nb_def)

#Calculating AUPRC
nb_prediction <-predict(nb_ldd_model, test_set_ldd_1,type ="prob")[,2]

nb_s1a<- nb_prediction[1:74]
nb_s2a <- nb_prediction[75:148]
nb_s3a <- nb_prediction[149:222]
nb_s4a <- nb_prediction[223:296]
nb_s5a <- nb_prediction[297:370]

nb_s_a<- join_scores(nb_s1a,nb_s2a,nb_s3a,nb_s4a,nb_s5a)
nb_mdat_ <- mmdata(scores=nb_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

nb_curves_ <- evalmod(scores=nb_prediction, labels= test_set_ldd_1$ldd_bin)
nb_curves1_ <- evalmod(nb_mdat_)

auc(nb_curves_)
auc(nb_curves1_)
auc_ci(nb_curves1_,alpha = 0.05, dtype = "normal")


#Running over-sampling
set.seed(1111111)
nb_ldd_model1 <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "naive_bayes",
  trControl = trControl1, metric= "F1")


#check for variable importance
nb_prediction1 <-predict(nb_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(nb_prediction1, test_set_ldd_1$ldd_bin, mode="everything")

confusion_matrix_nb <- matrix(c(83,65,36,186), nrow = 2, byrow = TRUE) 

#computing 95% CI for performance metrics
nb_data <- as.table(matrix(c(83,65,36,186), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
nb_rval <- epi.tests(nb_data, conf.level = 0.95,digits = 3)
print(nb_rval)

f1_nb_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nb_prediction1) %>%
  rename(obs=ldd_bin, pred=nb_prediction1) %>% mutate(obs=as.numeric(obs),
                                                     pred=as.numeric(pred))

bootout_nb1 <-boot(data=f1_nb_data1,
                  R=5000,
                  statistic=f1)

boot.ci(bootout_nb1,type="norm")
f1(f1_nb_data1)

#generating ROC Curve
roc_nb <- roc(test_set_ldd_1$ldd_bin,
              predict(nb_ldd_model1, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc3<-ggroc(roc_nb, colour="Red") +
  ggtitle(paste0("Naive Bayes ROC Curve","(AUC=", round(pROC::auc(roc_nb),digits=4), ")")) +
  theme_minimal()

roc3

## Now plot
roc3a <- as.ggplot(function() plot(roc_nb, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc3a <- roc3a +
  labs(title=paste0("Naive Bayes ROC Curve","(AUC=", round(pROC::auc(roc_nb),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc3a

pROC::auc(roc_nb)
ci.auc(roc_nb)

#Calculating AUPRC
nb_prediction1 <-predict(nb_ldd_model1, test_set_ldd_1,type ="prob")[,2]

nb_s1<- nb_prediction1[1:74]
nb_s2 <- nb_prediction1[75:148]
nb_s3 <- nb_prediction1[149:222]
nb_s4 <- nb_prediction1[223:296]
nb_s5 <- nb_prediction1[297:370]

nb_s<- join_scores(nb_s1,nb_s2,nb_s3,nb_s4,nb_s5)
nb_mdat <- mmdata(scores=nb_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

nb_curves <- evalmod(scores=nb_prediction1, labels= test_set_ldd_1$ldd_bin)
nb_curves1 <- evalmod(nb_mdat)

auc(nb_curves)
auc(nb_curves1)
auc_ci(nb_curves1,alpha = 0.05, dtype = "normal")

#Assessing Calibration

nb_prediction1a <-predict(nb_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(nb_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#-----------------------------------------------------------------------------#
#4.Logistic Regression Algorithm 

set.seed(7777777)
glm_ldd_model <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "glm",
  trControl = trControl,metric= "F1")

glm_prediction <-predict(glm_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(glm_prediction, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
glm_data_def <- as.table(matrix(c(61,23,58,228), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
glm_rval_def <- epi.tests(glm_data_def, conf.level = 0.95,digits = 3)
print(glm_rval_def)

f1_glm_data <- test_set_ldd_1 %>%
  select(ldd_bin) %>% cbind(glm_prediction) %>%
  rename(obs=ldd_bin, pred=glm_prediction) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_glm <-boot(data=f1_glm_data,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_glm,type="norm")
f1(f1_glm_data)

roc_glm_def <- roc(test_set_ldd_1$ldd_bin,
               predict(glm_ldd_model, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

pROC::auc(roc_glm_def)
ci.auc(roc_glm_def)

#Calculating AUPRC
glm_prediction <-predict(glm_ldd_model, test_set_ldd_1,type ="prob")[,2]

glm_s1a<- glm_prediction[1:74]
glm_s2a <- glm_prediction[75:148]
glm_s3a <- glm_prediction[149:222]
glm_s4a <- glm_prediction[223:296]
glm_s5a <- glm_prediction[297:370]

glm_s_a<- join_scores(glm_s1a,glm_s2a,glm_s3a,glm_s4a,glm_s5a)
glm_mdat_ <- mmdata(scores=glm_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

glm_curves_ <- evalmod(scores=glm_prediction, labels= test_set_ldd_1$ldd_bin)
glm_curves1_ <- evalmod(glm_mdat_)

auc(glm_curves_)
auc(glm_curves1_)
auc_ci(glm_curves1_,alpha = 0.05, dtype = "normal")


#running over-sampling
set.seed(1234567)
glm_ldd_model1 <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "glm",
  trControl = trControl1,metric= "F1")

glm_prediction1 <-predict(glm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(glm_prediction1, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_LR <- matrix(c(91,64,28,187), nrow = 2, byrow = TRUE) 

#computing 95% CI for performance metrics
glm_data <- as.table(matrix(c(91,64,28,187), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
glm_rval <- epi.tests(glm_data, conf.level = 0.95,digits = 3)
print(glm_rval)

f1_glm_data1 <- test_set_ldd_1 %>%
  select(ldd_bin) %>% cbind(glm_prediction1) %>%
  rename(obs=ldd_bin, pred=glm_prediction1) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_glm1 <-boot(data=f1_glm_data1,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_glm1,type="norm")
f1(f1_glm_data1)

#generating ROC Curve
roc_glm <- roc(test_set_ldd_1$ldd_bin,
               predict(glm_ldd_model1, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc4<-ggroc(roc_glm, colour="Red") +
  ggtitle(paste0("Logistic Regression ROC Curve","(AUC=", round(pROC::auc(roc_glm),digits=4), ")")) +
  theme_minimal()

roc4

## Now plot
roc4a <- as.ggplot(function() plot(roc_glm, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc4a <- roc4a +
  labs(title=paste0("Logistic Regression ROC Curve","(AUC=", round(pROC::auc(roc_glm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc4a

pROC::auc(roc_glm)
ci.auc(roc_glm)

#Calculating AUPRC
glm_prediction1 <-predict(glm_ldd_model1, test_set_ldd_1,type ="prob")[,2]

glm_s1<- glm_prediction1[1:74]
glm_s2 <- glm_prediction1[75:148]
glm_s3 <- glm_prediction1[149:222]
glm_s4 <- glm_prediction1[223:296]
glm_s5 <- glm_prediction1[297:370]

glm_s<- join_scores(glm_s1,glm_s2,glm_s3,glm_s4,glm_s5)
glm_mdat <- mmdata(scores=glm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

glm_curves <- evalmod(scores=glm_prediction1, labels= test_set_ldd_1$ldd_bin)
glm_curves1 <- evalmod(glm_mdat)

auc(glm_curves)
auc(glm_curves1)
auc_ci(glm_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_glm <- explain( model = glm_ldd_model1, data = X, y = Y,label = "Logistic Regression",type = "classification")

explainer_glm1 <- update_data(explainer_glm, data = X1, y = Y1)


resids_glm <- model_performance(explainer_glm)
plot(resids_glm)

mp_glm <- model_parts(explainer_glm1, type = "difference")
plot (mp_glm, show_boxplots = FALSE)

glm_shap <-predict_parts(explainer = explainer_glm,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

plot(glm_shap, show_boxplots = FALSE)

#Assessing Calibration

glm_prediction1a <-predict(glm_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(glm_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))



#-----------------------------------------------------------------------------#
#5. Run the SVM Algorithm 

set.seed(7777777)
svm_ldd_model <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "svmLinear",
                        trControl = trControl, preProcess = c("center","scale"),
                        probability=TRUE,
                        metric= "F1")

svm_prediction <-predict(svm_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
svm_data_def <- as.table(matrix(c(58,25,61,226), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
svm_rval_def <- epi.tests(svm_data_def, conf.level = 0.95,digits = 3)
print(svm_rval_def)

f1_svm_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(svm_prediction) %>%
  rename(obs=ldd_bin, pred=svm_prediction) %>% mutate(obs=as.numeric(obs),
                                                       pred=as.numeric(pred))

bootout_svm1 <-boot(data=f1_svm_data1,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_svm1,type="basic")
f1(f1_svm_data1)

roc_svm_def <- roc(test_set_ldd_1$ldd_bin,
               predict(svm_ldd_model, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

pROC::auc(roc_svm_def)
ci.auc(roc_svm_def)

#Calculating AUPRC
svm_prediction <-predict(svm_ldd_model, test_set_ldd_1,type ="prob")[,2]

svm_s1a<- svm_prediction[1:74]
svm_s2a <- svm_prediction[75:148]
svm_s3a <- svm_prediction[149:222]
svm_s4a <- svm_prediction[223:296]
svm_s5a <- svm_prediction[297:370]

svm_s_a<- join_scores(svm_s1a,svm_s2a,svm_s3a,svm_s4a,svm_s5a)
svm_mdat_ <- mmdata(scores=svm_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

svm_curves_ <- evalmod(scores=svm_prediction, labels= test_set_ldd_1$ldd_bin)
svm_curves1_ <- evalmod(svm_mdat_)

auc(svm_curves_)
auc(svm_curves1_)
auc_ci(svm_curves1_,alpha = 0.05, dtype = "normal")

#implementing Over-sampling
set.seed(1111111)
svm_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "svmLinear",
                        trControl = trControl1, preProcess = c("center","scale"),
                        probability=TRUE,
                        metric= "F1")


svm_prediction1 <-predict(svm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction1, test_set_ldd_1$ldd_bin)

#Parameter tuning
# Grid search to fine tune SVM
grid <- expand.grid(sigma = c(.01, .015, 0.2),
                    C = c(0.75, 0.9, 1, 1.1, 1.25))

set.seed(1111111)
svm_ldd_model4a <- train(ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "svmRadial",
                         trControl = trControl1, preProcess = c("center","scale"),
                         probability=TRUE,
                         tuneGrid = grid,
                         metric= "F1")

svm_prediction4a <-predict(svm_ldd_model4a, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction4a, test_set_ldd_1$ldd_bin)

# #saving model
# saveRDS(svm_ldd_model4a, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_svm.rda")
# saveRDS(svm_ldd_model4a, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_svm.rds")
# 
# #loading model 
# svm_ldd_model4a= readRDS("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_svm.rda")
# svm_ldd_model4a= readRDS("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Longer duration Diarrhea/Data/LDD_svm.rds")
#  
grid1 <- expand.grid(C = c(0.75, 0.9, 1, 1.1, 1.25))

set.seed(1111111)
svm_ldd_model4b <- train(ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "svmLinear",
                         trControl = trControl1, preProcess = c("center","scale"),
                         probability=TRUE,
                         tuneGrid = grid1,
                         metric= "F1")

svm_prediction4b <-predict(svm_ldd_model4b, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction4b, test_set_ldd_1$ldd_bin)

#SVMRadial gives a more balanced result 
svm_prediction4a <-predict(svm_ldd_model4a, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction4a, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_svm <- matrix(c(88,59,31,192), nrow = 2, byrow = TRUE)

#computing 95% CI for performance metrics
svm_data <- as.table(matrix(c(88,59,31,192), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
svm_rval <- epi.tests(svm_data, conf.level = 0.95,digits = 3)
print(svm_rval)

f1_svm_data2 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(svm_prediction4a) %>%
  rename(obs=ldd_bin, pred=svm_prediction4a) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_svm2 <-boot(data=f1_svm_data2,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_svm2,type="norm")
f1(f1_svm_data2)

#generating ROC Curve
roc_svm <- roc(test_set_ldd_1$ldd_bin,
               predict(svm_ldd_model4a, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
roc5<-ggroc(roc_svm, colour="Red") +
  ggtitle(paste0("SVM ROC Curve","(AUC=", round(pROC::auc(roc_svm),digits=4), ")")) +
  theme_minimal()

roc5

## Now plot
roc5a <- as.ggplot(function() plot(roc_svm, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc5a <- roc5a +
  labs(title=paste0("SVM ROC Curve","(AUC=", round(pROC::auc(roc_svm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc5a

pROC::auc(roc_svm)
ci.auc(roc_svm)

#Calculating AUPRC
svm_prediction4a <-predict(svm_ldd_model4a, test_set_ldd_1,type ="prob")[,2]

svm_s1<- svm_prediction4a[1:74]
svm_s2 <- svm_prediction4a[75:148]
svm_s3 <- svm_prediction4a[149:222]
svm_s4 <- svm_prediction4a[223:296]
svm_s5 <- svm_prediction4a[297:370]

svm_s<- join_scores(svm_s1,svm_s2,svm_s3,svm_s4,svm_s5)
svm_mdat <- mmdata(scores=svm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

svm_curves <- evalmod(scores=svm_prediction4a, labels= test_set_ldd_1$ldd_bin)
svm_curves1 <- evalmod(svm_mdat)

auc(svm_curves)
auc(svm_curves1)
auc_ci(svm_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_svm <- explain( model = svm_ldd_model4a, data = X, y = Y,label = "SVM",type = "classification")

explainer_svm1 <- update_data(explainer_svm, data = X1, y = Y1)


resids_svm <- model_performance(explainer_svm1)
plot(resids_svm)

mp_svm <- model_parts(explainer_svm1, type = "difference")
plot (mp_svm, show_boxplots = FALSE)

svm_shap <-predict_parts(explainer = explainer_svm,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

svm_shap1 <- plot(svm_shap, show_boxplots = FALSE)

svm_shap1
# Computing business value of model using modelplotr

# scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("train_set_ldd_1","test_set_ldd_1"),
#                                                dataset_labels = list("train data","test data"),
#                                                models = list("svm_ldd_model4a"),  
#                                                model_labels = list("SVM"), 
#                                                target_column="ldd_bin",
#                                                ntiles = 100)
# 
# plot_input <- plotting_scope(prepared_input = scores_and_ntiles)
# 
# #cummulative gains 
# plot_cumgains(data = plot_input)
# 
# #Cumulative lift
# plot_cumlift(data = plot_input)
# 
# #Response plot
# plot_response(data = plot_input)
# 
# #Cumulative response plot
# plot_cumresponse(data = plot_input)
# 
# #multiple plots
# plot_multiplot(data = plot_input)


#Assessing Calibration

svm_prediction4a_ <-predict(svm_ldd_model4a, test_set_ldd_1,type = "prob")
val.prob(svm_prediction4a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))

#Comparing TP and FN among LDD cases
#characteristics of LDD not predicted accurately
test_set_ldd_1a <- test_set_ldd_1 %>%
  cbind(svm_prediction4a) %>%
  filter(ldd_bin=="Yes") #& test_set_ldd_1a$svm_prediction4a=="NO"


#tab1(test_set_ldd_1a$VESIKARI_SCORE, cum.percent = TRUE)
CrossTable(test_set_ldd_1a$AGEGROUP, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$breast_feed, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$F4A_ANY_VOMIT, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$F4A_FREQ_VOMIT1, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$rotavax1, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$F4B_SKIN1, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$F4A_DAILY_MAX1, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

CrossTable(test_set_ldd_1a$VESIKARI_SCORE, test_set_ldd_1a$svm_prediction4a, prop.c=TRUE,
           prop.r=FALSE, prop.chisq=FALSE, prop.t=FALSE, chisq=TRUE, fisher=TRUE)

test_set_ldd_1a_ <- test_set_ldd_1a %>% filter(svm_prediction4a=="Yes") 
quantile(test_set_ldd_1a_$F4A_DRH_DAYS)

test_set_ldd_1a_b <- test_set_ldd_1a %>% filter(svm_prediction4a=="NO") 
quantile(test_set_ldd_1a_b$F4A_DRH_DAYS)

wilcox.test(F4A_DRH_DAYS ~ svm_prediction4a, data = test_set_ldd_1a,
            exact = FALSE)

quantile(test_set_ldd_1a_$resp_rate)
quantile(test_set_ldd_1a_b$resp_rate)
wilcox.test(resp_rate ~ svm_prediction4a, data = test_set_ldd_1a,
            exact = FALSE)

quantile(test_set_ldd_1a_$F4A_DAYS_VOMIT2)
quantile(test_set_ldd_1a_b$F4A_DAYS_VOMIT2)
wilcox.test(F4A_DAYS_VOMIT2 ~ svm_prediction4a, data = test_set_ldd_1a,
            exact = FALSE)
#-----------------------------------------------------------------------------#
#6. Run the kNN Algorithm 

set.seed(7777777)
knn_ldd_model <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "knn",
                        trControl = trControl,metric= "F1")

knn_prediction <-predict(knn_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(knn_prediction, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
knn_data_def <- as.table(matrix(c(52,28,67,223), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
knn_rval_def <- epi.tests(knn_data_def, conf.level = 0.95,digits = 3)
print(knn_rval_def)

f1_knn_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(knn_prediction) %>%
  rename(obs=ldd_bin, pred=knn_prediction) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_knn <-boot(data=f1_knn_data,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_knn,type="norm")
f1(f1_knn_data)

#generating ROC Curve
roc_knn_def <- roc(test_set_ldd_1$ldd_bin,
               predict(knn_ldd_model, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

pROC::auc(roc_knn_def)
ci.auc(roc_knn_def)

#Calculating AUPRC
knn_prediction <-predict(knn_ldd_model, test_set_ldd_1,type ="prob")[,2]

knn_s1a<- knn_prediction[1:74]
knn_s2a <- knn_prediction[75:148]
knn_s3a <- knn_prediction[149:222]
knn_s4a <- knn_prediction[223:296]
knn_s5a <- knn_prediction[297:370]

knn_s_a<- join_scores(knn_s1a,knn_s2a,knn_s3a,knn_s4a,knn_s5a)
knn_mdat_ <- mmdata(scores=knn_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

knn_curves_ <- evalmod(scores=knn_prediction, labels= test_set_ldd_1$ldd_bin)
knn_curves1_ <- evalmod(knn_mdat_)

auc(knn_curves_)
auc(knn_curves1)
auc_ci(knn_curves1_,alpha = 0.05, dtype = "normal")

#Running over-sampling
set.seed(1111111)
knn_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "knn",
                        trControl = trControl1,metric= "F1")


#check for variable importance
knn_prediction1 <-predict(knn_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(knn_prediction1, test_set_ldd_1$ldd_bin)



#Parameter tuning
set.seed(1111111)
knn_ldd_model1a <- train(ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "knn",
                         trControl = trControl1, metric= "F1",
                         tuneGrid = data.frame(k = seq(11,85,by = 2)))

knn_prediction1a <-predict(knn_ldd_model1a, test_set_ldd_1,type = "raw")
confusionMatrix(knn_prediction1a, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_knn <- matrix(c(87,70,32,181), nrow = 2, byrow = TRUE)

#computing 95% CI for performance metrics
knn_data <- as.table(matrix(c(87,70,32,181), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
knn_rval <- epi.tests(knn_data, conf.level = 0.95,digits = 3)
print(knn_rval)

#F1 95%CI
f1_knn_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(knn_prediction1a) %>%
  rename(obs=ldd_bin, pred=knn_prediction1a) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_knn1 <-boot(data=f1_knn_data1,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_knn1,type="norm")
f1(f1_knn_data1)

#generating ROC Curve
roc_knn <- roc(test_set_ldd_1$ldd_bin,
               predict(knn_ldd_model1a, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
roc6<-ggroc(roc_knn, colour="Red") +
  ggtitle(paste0("KNN ROC Curve","(AUC=", round(pROC::auc(roc_knn),digits=4), ")")) +
  theme_minimal()

roc6

## Now plot
roc6a <- as.ggplot(function() plot(roc_knn, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc6a <- roc6a +
  labs(title=paste0("KNN ROC Curve","(AUC=", round(pROC::auc(roc_knn),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc6a

pROC::auc(roc_knn)
ci.auc(roc_knn)


#Calculating AUPRC
knn_prediction1a <-predict(knn_ldd_model1a, test_set_ldd_1,type ="prob")[,2]

knn_s1<- knn_prediction1a[1:74]
knn_s2 <- knn_prediction1a[75:148]
knn_s3 <- knn_prediction1a[149:222]
knn_s4 <- knn_prediction1a[223:296]
knn_s5 <- knn_prediction1a[297:370]

knn_s<- join_scores(knn_s1,knn_s2,knn_s3,knn_s4,knn_s5)
knn_mdat <- mmdata(scores=knn_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

knn_curves <- evalmod(scores=knn_prediction1a, labels= test_set_ldd_1$ldd_bin)
knn_curves1 <- evalmod(knn_mdat)

auc(knn_curves)
auc(knn_curves1)
auc_ci(knn_curves1,alpha = 0.05, dtype = "normal")

#Assessing Calibration

knn_prediction1a_ <-predict(knn_ldd_model1a, test_set_ldd_1,type = "prob")
val.prob(knn_prediction1a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#------------------------------------------------------------------------------#
#Run the nueralnet Algorithm 

set.seed(77777777)
ann_ldd_model <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "nnet",
                        trControl = trControl,metric= "F1")

nn_prediction <-predict(ann_ldd_model, test_set_ldd_1,type = "raw")
confusionMatrix(nn_prediction, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
nn_data_def <- as.table(matrix(c(62,24,57,227), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
nn_rval_def <- epi.tests(nn_data_def, conf.level = 0.95,digits = 3)
print(nn_rval_def)

f1_nn_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nn_prediction) %>%
  rename(obs=ldd_bin, pred=nn_prediction) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_nn <-boot(data=f1_nn_data,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_nn,type="basic")
f1(f1_nn_data)

roc_nn_def <- roc(test_set_ldd_1$ldd_bin,
              predict(ann_ldd_model, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))
pROC::auc(roc_nn_def)
ci.auc(roc_nn_def)

#PRAUC
nn_prediction <-predict(ann_ldd_model, test_set_ldd_1,type ="prob")[,2]

nn_s1a<- nn_prediction[1:74]
nn_s2a <- nn_prediction[75:148]
nn_s3a <- nn_prediction[149:222]
nn_s4a <- nn_prediction[223:296]
nn_s5a <- nn_prediction[297:370]

nn_s_a<- join_scores(nn_s1a,nn_s2a,nn_s3a,nn_s4a,nn_s5a)
nn_mdat_ <- mmdata(scores=nn_s_a, labels= rf_l, modnames=c('m1'), dsids=1:5)

nn_curves_ <- evalmod(scores=nn_prediction, labels= test_set_ldd_1$ldd_bin)
nn_curves1_ <- evalmod(nn_mdat_)

auc(nn_curves_)
auc(nn_curves1_)
auc_ci(nn_curves1_,alpha = 0.05, dtype = "normal")

#Running over-sampling
set.seed(7777777)
ann_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "nnet",
                        trControl = trControl1,metric= "F1")


#check for variable importance
nn_prediction1 <-predict(ann_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(nn_prediction1, test_set_ldd_1$ldd_bin)


#Parameter Tuning
set.seed(7777777)
ann_ldd_model1a <- train(ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "nnet",
                         trControl = trControl1,metric= "F1",
                         activation=c("Maxout","MaxoutWithDropout"),
                         l1 = c(0, 0.00001, 0.0001), 
                         l2 = c(0, 0.00001, 0.0001),
                         hidden=list(c(5, 5, 5, 5, 5), c(10, 10, 10, 10), c(50, 50, 50), c(100, 100, 100))
)
print(ann_ldd_model1a)

nn_prediction1a <-predict(ann_ldd_model1a, test_set_ldd_1,type = "raw")
confusionMatrix(nn_prediction1a, test_set_ldd_1$ldd_bin)

confusionMatrix(nn_prediction1, test_set_ldd_1$ldd_bin, mode='everything')

confusion_matrix_ann <- matrix(c(90,63,29,188), nrow = 2, byrow = TRUE)

#computing 95% CI for performance metrics
nn_data <- as.table(matrix(c(90,63,29,188), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
nn_rval <- epi.tests(nn_data, conf.level = 0.95,digits = 3)
print(nn_rval)

f1_nn_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nn_prediction1) %>%
  rename(obs=ldd_bin, pred=nn_prediction1) %>% mutate(obs=as.numeric(obs),
                                                     pred=as.numeric(pred))

bootout_nn1 <-boot(data=f1_nn_data1,
                  R=5000,
                  statistic=f1)

boot.ci(bootout_nn1,type="norm")
f1(f1_nn_data1)

#generating ROC Curve
roc_nn <- roc(test_set_ldd_1$ldd_bin,
              predict(ann_ldd_model1, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc7<-ggroc(roc_nn, colour="Red") +
  ggtitle(paste0("ANN ROC Curve","(AUC=", round(pROC::auc(roc_nn),digits=4), ")")) +
  theme_minimal()

roc7

## Now plot
roc7a <- as.ggplot(function() plot(roc_nn, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE))

roc7a <- roc7a +
  labs(title=paste0("ANN ROC Curve","(AUC=", round(pROC::auc(roc_nn),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc7a

pROC::auc(roc_nn)
ci.auc(roc_nn)

#Calculating AUPRC
nn_prediction1 <-predict(ann_ldd_model1, test_set_ldd_1,type ="prob")[,2]

nn_s1<- nn_prediction1[1:74]
nn_s2 <- nn_prediction1[75:148]
nn_s3 <- nn_prediction1[149:222]
nn_s4 <- nn_prediction1[223:296]
nn_s5 <- nn_prediction1[297:370]

nn_s<- join_scores(nn_s1,nn_s2,nn_s3,nn_s4,nn_s5)
nn_mdat <- mmdata(scores=nn_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

nn_curves <- evalmod(scores=nn_prediction1, labels= test_set_ldd_1$ldd_bin)
nn_curves1 <- evalmod(nn_mdat)

auc(nn_curves)
auc(nn_curves1)
auc_ci(nn_curves1,alpha = 0.05, dtype = "normal")

#computing shapley values using DALEX
explainer_ann <- explain( model = ann_ldd_model1, data = X, y = Y,label = "ANN",type = "classification")

explainer_ann1 <- update_data(explainer_ann, data = X1, y = Y1)


resids_ann <- model_performance(explainer_ann1)
plot(resids_ann)

mp_ann <- model_parts(explainer_ann1)
plot (mp_ann, show_boxplots = FALSE)

ann_shap <-predict_parts(explainer = explainer_ann,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

ann_shap1 <- plot(ann_shap, show_boxplots = FALSE)
ann_shap1

#Assessing Calibration

nn_prediction1a_ <-predict(ann_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(nn_prediction1a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#Merging ROC Curves into one graph

grid.arrange(roc1,roc2,roc3,roc4,roc5,roc6,roc7, ncol = 3 )

grid.arrange(roc1a,roc2a,roc3a,roc4a,roc5a,roc6a,roc7a, ncol = 3)

grid.arrange(rf_shap1,svm_shap1,gbm_shap1,ann_shap1,  ncol=2)

##Comparing model performance to Logistic Regression

mcnemar_test_result1 <- mcnemar.test(confusion_matrix_RF, confusion_matrix_LR)
mcnemar_test_result2 <- mcnemar.test(confusion_matrix_gbm, confusion_matrix_LR)
mcnemar_test_result3 <- mcnemar.test(confusion_matrix_nb, confusion_matrix_LR)
mcnemar_test_result4 <- mcnemar.test(confusion_matrix_svm, confusion_matrix_LR)
mcnemar_test_result5 <- mcnemar.test(confusion_matrix_knn, confusion_matrix_LR)
mcnemar_test_result6 <- mcnemar.test(confusion_matrix_ann, confusion_matrix_LR)

# Print the result
print(mcnemar_test_result1)
print(mcnemar_test_result2)
print(mcnemar_test_result3)
print(mcnemar_test_result4)
print(mcnemar_test_result5)
print(mcnemar_test_result6)


model_performance <- data.frame(
  Model = c("RF", "GBM", "NB", "LR", "SVM", "KNN","ANN"),
  Sensitivity = c(0.807, 0.723, 0.697,0.765,0.739, 0.731, 0.756),  # Performance metric values for each model
  SPecificity = c(0.741, 0.765, 0.741, 0.745, 0.721, 0.605, 0.749),
  PPV = c(0.596, 0.593, 0.561, 0.587, 0.599, 0.554, 0.588),
  NPV = c(0.89, 0.853, 0.838, 0.870, 0.861, 0.85, 0.866),
  F1 = c(0.686, 0.652, 0.838, 0.664, 0.662, 0.63, 0.661),
  AUC=c(0.830, 0.811, 0.773, 0.805, 0.82, 0.797, 0.815)
)

data_long <- model_performance %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")


# Perform Friedman test (replace with your actual Friedman test results)
friedman_result <- friedman.test(data_long$Value, data_long$Model, data_long$Metric)
print(friedman_result)

# Conduct Nemenyi post-hoc test
nemenyi_result <- frdAllPairsNemenyiTest(data_long$Value, data_long$Model, data_long$Metric)
print(nemenyi_result)

plot(nemenyi_result)


##
##
##
#Scenario 2: Excluding pre-enrolment diarrhea days from the prediction of LDD
##
##

set.seed(1111111)
test_index_ldd_1 <- createDataPartition(LDD_data3c$ldd_bin, times = 1, p = 0.75, list = FALSE)
train_set_ldd_1  <- LDD_data3c[test_index_ldd_1, ]
test_set_ldd_1 <- LDD_data3c[-test_index_ldd_1, ]

X<- train_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y<- train_set_ldd_1 %>% dplyr::select(ldd_bin)  %>% mutate(ldd_bin=as.numeric(ldd_bin))
X1 <- test_set_ldd_1 %>% dplyr::select(-ldd_bin)
Y1 <- test_set_ldd_1 %>% dplyr::select(ldd_bin) %>% mutate(ldd_bin=as.numeric(ldd_bin))




##---------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------##
#Random Forest
#Running Over-sampling

set.seed(3333333)
rf_default_M1a <- train(ldd_bin ~ ., method = "rf", data=train_set_ldd_1, metric= "F1",
                        trControl = trControl1,threshold = 0.3)

rf_prediction_M1a <-predict(rf_default_M1a, test_set_ldd_1,type = "raw")
confusionMatrix(rf_prediction_M1a, test_set_ldd_1$ldd_bin, mode='everything')

varImp(rf_default_M1a)


#computing 95% CI for performance metrics
rf_data <- as.table(matrix(c(91,117,28,134), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
rf_rval <- epi.tests(rf_data, conf.level = 0.95,digits = 3)
print(rf_rval)

#computing 95% CI for F1-score
f1_rf_data1 <- test_set_ldd_1 %>%
  dplyr:: select(ldd_bin) %>% cbind(rf_prediction_M1a) %>%
  rename(obs=ldd_bin, pred=rf_prediction_M1a) %>% mutate(obs=as.numeric(obs),
                                                         pred=as.numeric(pred))

bootout_rf<-boot(data=f1_rf_data1,
                 R=5000,
                 statistic=f1)

boot.ci(bootout_rf,type="norm")
f1(f1_rf_data1)


#generating ROC Curve
roc_rf <- roc(test_set_ldd_1$ldd_bin,
              predict(rf_default_M1a, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))


roc1<-ggroc(roc_rf, colour="Red") +
  ggtitle(paste0("Random Forest ROC Curve","(AUC=", round(pROC::auc(roc_rf),digits=4), ")")) +
  theme_minimal()

roc1

## Now plot
roc1a <- as.ggplot(function() plot(roc_rf, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc1a <- roc1a +
  labs(title=paste0("Random Forest ROC Curve","(AUC=", round(pROC::auc(roc_rf),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc1a

pROC::auc(roc_rf)
ci.auc(roc_rf)


#Calculating AUPRC
rf_prediction_M1a <-predict(rf_default_M1a, test_set_ldd_1,type ="prob")[,2]

rf_s1<- rf_prediction_M1a[1:74]
rf_s2 <- rf_prediction_M1a[75:148]
rf_s3 <- rf_prediction_M1a[149:222]
rf_s4 <- rf_prediction_M1a[223:296]
rf_s5 <- rf_prediction_M1a[297:370]


rf_l1 <- test_set_ldd_1$ldd_bin[1:74]
rf_l2 <- test_set_ldd_1$ldd_bin[75:148]
rf_l3 <- test_set_ldd_1$ldd_bin[149:222]
rf_l4 <- test_set_ldd_1$ldd_bin[223:296]
rf_l5 <- test_set_ldd_1$ldd_bin[297:370]

rf_s<- join_scores(rf_s1,rf_s2,rf_s3,rf_s4,rf_s5)
rf_l <- join_labels(rf_l1,rf_l2,rf_l3,rf_l4,rf_l5)

rfmdat <- mmdata(scores=rf_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

rf_curves <- evalmod(scores=rf_prediction_M1a, labels= test_set_ldd_1$ldd_bin)
rf_curves1 <- evalmod(rfmdat)

auc(rf_curves)
auc(rf_curves1)
auc_ci(rf_curves1,alpha = 0.05, dtype = "normal")



#computing shapley values using DALEX
explainer_rf <- explain( model = rf_default_M1a, data = X, y = Y,label = "Random forest",type = "classification")

explainer_rf_1 <- update_data(explainer_rf, data = X1, y = Y1)


resids_rf <- model_performance(explainer_rf_1)
plot(resids_rf)

mp_rf <- model_parts(explainer_rf_1, type = "difference")
plot (mp_rf, show_boxplots = FALSE)
#axis(2,labels=format(mp_rf,scientific=FALSE))



rf_shap <-predict_parts(explainer = explainer_rf,
                        new_observation = X1,
                        type = "shap",
                        B = 10 #number of reorderings - start small
)

rf_shap1 <-  plot(rf_shap, show_boxplots = FALSE)
rf_shap1


#Assessing model calibration
rf_prediction_M1a_ <-predict(rf_default_M1a, test_set_ldd_1,type = "prob")

test_set_ldd_1_ <- test_set_ldd_1 %>% 
  mutate(ldd_bin=as.numeric(test_set_ldd_1$ldd_bin)) %>%
  mutate(ldd_bin=if_else(ldd_bin==2,0,ldd_bin))


val.prob(rf_prediction_M1a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#___________________________________________________________________________________________________#
#2.Gradient Boosting Algorithm
#running over-sampling
set.seed(1111111)
gbm_ldd_model1 <- train( ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "xgbTree", 
                         trControl = trControl1,metric= "F1") #shrinkage = 0.01,

gbm_prediction1 <-predict(gbm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(gbm_prediction1, test_set_ldd_1$ldd_bin, mode='everything')



#computing 95% CI for performance metrics
gbm_data <- as.table(matrix(c(88,104,31,147), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
gbm_rval <- epi.tests(gbm_data, conf.level = 0.95,digits = 3)
print(gbm_rval)

#computing 95% CI for F1-score
f1_gbm_data <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(gbm_prediction1) %>%
  rename(obs=ldd_bin, pred=gbm_prediction1) %>% mutate(obs=as.numeric(obs),
                                                       pred=as.numeric(pred))

bootout_gbm1 <-boot(data=f1_gbm_data,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_gbm1,type="norm")
f1(f1_gbm_data)

#generating ROC Curve
roc_gbm <- roc(test_set_ldd_1$ldd_bin,
               predict(gbm_ldd_model1, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc2<-ggroc(roc_gbm, colour="Red") +
  ggtitle(paste0("Gradient Boosting ROC Curve","(AUC=", round(pROC::auc(roc_gbm),digits=4), ")")) +
  theme_minimal()

roc2

## Now plot
roc2a <- as.ggplot(function() plot(roc_gbm, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc2a <- roc2a +
  labs(title=paste0("Gradient Boosting ROC Curve","(AUC=", round(pROC::auc(roc_gbm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc2a

pROC::auc(roc_gbm)
ci.auc(roc_gbm)

#Calculating AUPRC
gbm_prediction1 <-predict(gbm_ldd_model1, test_set_ldd_1,type ="prob")[,2]


gbm_s1<- gbm_prediction1[1:74]
gbm_s2 <- gbm_prediction1[75:148]
gbm_s3 <- gbm_prediction1[149:222]
gbm_s4 <- gbm_prediction1[223:296]
gbm_s5 <- gbm_prediction1[297:370]

gbm_s<- join_scores(gbm_s1,gbm_s2,gbm_s3,gbm_s4,gbm_s5)
gbm_mdat <- mmdata(scores=gbm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

gbm_curves <- evalmod(scores=gbm_prediction1, labels= test_set_ldd_1$ldd_bin)
gbm_curves1 <- evalmod(gbm_mdat)

auc(gbm_curves)
auc(gbm_curves1)
auc_ci(gbm_curves1,alpha = 0.05, dtype = "normal")

#computing shapley values using DALEX
explainer_gbm <- explain( model = gbm_ldd_model1, data = X, y = Y,label = "Gradient Boosting",type = "classification")

explainer_gbm1 <- update_data(explainer_gbm, data = X1, y = Y1)


resids_gbm <- model_performance(explainer_gbm1)
plot(resids_gbm)

mp_gbm <- model_parts(explainer_gbm1, type = "difference")
plot (mp_gbm, show_boxplots = FALSE)


gbm_shap <-predict_parts(explainer = explainer_gbm,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

gbm_shap1 <- plot(gbm_shap, show_boxplots = FALSE)
gbm_shap1

#Assessing Calibration

gbm_prediction1a <-predict(gbm_ldd_model1, test_set_ldd_1,type = "prob")

val.prob(gbm_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))

#----------------------------------------------------------------------------#
#3.Naive Bayes Algorithm -hyper parameter tuning not recommended for NB
#Running over-sampling
set.seed(1111111)
nb_ldd_model1 <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "naive_bayes",
  trControl = trControl1, metric= "F1")


#check for variable importance
nb_prediction1 <-predict(nb_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(nb_prediction1, test_set_ldd_1$ldd_bin, mode="everything")


#computing 95% CI for performance metrics
nb_data <- as.table(matrix(c(96,163,23,88), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
nb_rval <- epi.tests(nb_data, conf.level = 0.95,digits = 3)
print(nb_rval)

f1_nb_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nb_prediction1) %>%
  rename(obs=ldd_bin, pred=nb_prediction1) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_nb1 <-boot(data=f1_nb_data1,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_nb1,type="norm")
f1(f1_nb_data1)

#generating ROC Curve
roc_nb <- roc(test_set_ldd_1$ldd_bin,
              predict(nb_ldd_model1, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc3<-ggroc(roc_nb, colour="Red") +
  ggtitle(paste0("Naive Bayes ROC Curve","(AUC=", round(pROC::auc(roc_nb),digits=4), ")")) +
  theme_minimal()

roc3

## Now plot
roc3a <- as.ggplot(function() plot(roc_nb, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc3a <- roc3a +
  labs(title=paste0("Naive Bayes ROC Curve","(AUC=", round(pROC::auc(roc_nb),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc3a

pROC::auc(roc_nb)
ci.auc(roc_nb)

#Calculating AUPRC
nb_prediction1 <-predict(nb_ldd_model1, test_set_ldd_1,type ="prob")[,2]

nb_s1<- nb_prediction1[1:74]
nb_s2 <- nb_prediction1[75:148]
nb_s3 <- nb_prediction1[149:222]
nb_s4 <- nb_prediction1[223:296]
nb_s5 <- nb_prediction1[297:370]

nb_s<- join_scores(nb_s1,nb_s2,nb_s3,nb_s4,nb_s5)
nb_mdat <- mmdata(scores=nb_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

nb_curves <- evalmod(scores=nb_prediction1, labels= test_set_ldd_1$ldd_bin)
nb_curves1 <- evalmod(nb_mdat)

auc(nb_curves)
auc(nb_curves1)
auc_ci(nb_curves1,alpha = 0.05, dtype = "normal")

#Assessing Calibration

nb_prediction1a <-predict(nb_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(nb_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#-----------------------------------------------------------------------------#
#4.Logistic Regression Algorithm 
#running over-sampling
set.seed(1234567)
glm_ldd_model1 <- train(
  ldd_bin ~ ., 
  data = train_set_ldd_1, method = "glm",
  trControl = trControl1,metric= "F1")

glm_prediction1 <-predict(glm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(glm_prediction1, test_set_ldd_1$ldd_bin, mode='everything')


#computing 95% CI for performance metrics
glm_data <- as.table(matrix(c(92,119,27,132), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
glm_rval <- epi.tests(glm_data, conf.level = 0.95,digits = 3)
print(glm_rval)

f1_glm_data1 <- test_set_ldd_1 %>%
  select(ldd_bin) %>% cbind(glm_prediction1) %>%
  rename(obs=ldd_bin, pred=glm_prediction1) %>% mutate(obs=as.numeric(obs),
                                                       pred=as.numeric(pred))

bootout_glm1 <-boot(data=f1_glm_data1,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_glm1,type="norm")
f1(f1_glm_data1)

#generating ROC Curve
roc_glm <- roc(test_set_ldd_1$ldd_bin,
               predict(glm_ldd_model1, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc4<-ggroc(roc_glm, colour="Red") +
  ggtitle(paste0("Logistic Regression ROC Curve","(AUC=", round(pROC::auc(roc_glm),digits=4), ")")) +
  theme_minimal()

roc4

## Now plot
roc4a <- as.ggplot(function() plot(roc_glm, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc4a <- roc4a +
  labs(title=paste0("Logistic Regression ROC Curve","(AUC=", round(pROC::auc(roc_glm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc4a

pROC::auc(roc_glm)
ci.auc(roc_glm)

#Calculating AUPRC
glm_prediction1 <-predict(glm_ldd_model1, test_set_ldd_1,type ="prob")[,2]

glm_s1<- glm_prediction1[1:74]
glm_s2 <- glm_prediction1[75:148]
glm_s3 <- glm_prediction1[149:222]
glm_s4 <- glm_prediction1[223:296]
glm_s5 <- glm_prediction1[297:370]

glm_s<- join_scores(glm_s1,glm_s2,glm_s3,glm_s4,glm_s5)
glm_mdat <- mmdata(scores=glm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

glm_curves <- evalmod(scores=glm_prediction1, labels= test_set_ldd_1$ldd_bin)
glm_curves1 <- evalmod(glm_mdat)

auc(glm_curves)
auc(glm_curves1)
auc_ci(glm_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_glm <- explain( model = glm_ldd_model1, data = X, y = Y,label = "Logistic Regression",type = "classification")

explainer_glm1 <- update_data(explainer_glm, data = X1, y = Y1)


resids_glm <- model_performance(explainer_glm)
plot(resids_glm)

mp_glm <- model_parts(explainer_glm1, type = "difference")
plot (mp_glm, show_boxplots = FALSE)

glm_shap <-predict_parts(explainer = explainer_glm,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

plot(glm_shap, show_boxplots = FALSE)

#Assessing Calibration

glm_prediction1a <-predict(glm_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(glm_prediction1a$Yes, as.numeric(test_set_ldd_1_$ldd_bin))



#-----------------------------------------------------------------------------#
#5. Run the SVM Algorithm 
#implementing Over-sampling
set.seed(7777777)
svm_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "svmLinear",
                        trControl = trControl1, preProcess = c("center","scale"),
                        probability=TRUE,
                        metric= "F1")


svm_prediction1 <-predict(svm_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(svm_prediction1, test_set_ldd_1$ldd_bin, mode='everything')


#computing 95% CI for performance metrics
svm_data <- as.table(matrix(c(102,167,17,84), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
svm_rval <- epi.tests(svm_data, conf.level = 0.95,digits = 3)
print(svm_rval)

f1_svm_data2 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(svm_prediction1) %>%
  rename(obs=ldd_bin, pred=svm_prediction1) %>% mutate(obs=as.numeric(obs),
                                                        pred=as.numeric(pred))

bootout_svm2 <-boot(data=f1_svm_data2,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_svm2,type="norm")
f1(f1_svm_data2)

#generating ROC Curve
roc_svm <- roc(test_set_ldd_1$ldd_bin,
               predict(svm_ldd_model1, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
roc5<-ggroc(roc_svm, colour="Red") +
  ggtitle(paste0("SVM ROC Curve","(AUC=", round(pROC::auc(roc_svm),digits=4), ")")) +
  theme_minimal()

roc5

## Now plot
roc5a <- as.ggplot(function() plot(roc_svm, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc5a <- roc5a +
  labs(title=paste0("SVM ROC Curve","(AUC=", round(pROC::auc(roc_svm),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc5a

pROC::auc(roc_svm)
ci.auc(roc_svm)

#Calculating AUPRC
svm_prediction4a <-predict(svm_ldd_model1, test_set_ldd_1,type ="prob")[,2]

svm_s1<- svm_prediction4a[1:74]
svm_s2 <- svm_prediction4a[75:148]
svm_s3 <- svm_prediction4a[149:222]
svm_s4 <- svm_prediction4a[223:296]
svm_s5 <- svm_prediction4a[297:370]

svm_s<- join_scores(svm_s1,svm_s2,svm_s3,svm_s4,svm_s5)
svm_mdat <- mmdata(scores=svm_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

svm_curves <- evalmod(scores=svm_prediction4a, labels= test_set_ldd_1$ldd_bin)
svm_curves1 <- evalmod(svm_mdat)

auc(svm_curves)
auc(svm_curves1)
auc_ci(svm_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_svm <- explain( model = svm_ldd_model4a, data = X, y = Y,label = "SVM",type = "classification")

explainer_svm1 <- update_data(explainer_svm, data = X1, y = Y1)


resids_svm <- model_performance(explainer_svm1)
plot(resids_svm)

mp_svm <- model_parts(explainer_svm1, type = "difference")
plot (mp_svm, show_boxplots = FALSE)

svm_shap <-predict_parts(explainer = explainer_svm,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

svm_shap1 <- plot(svm_shap, show_boxplots = FALSE)

svm_shap1
# Computing business value of model using modelplotr

scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("train_set_ldd_1","test_set_ldd_1"),
                                               dataset_labels = list("train data","test data"),
                                               models = list("svm_ldd_model4a"),  
                                               model_labels = list("SVM"), 
                                               target_column="ldd_bin",
                                               ntiles = 100)

plot_input <- plotting_scope(prepared_input = scores_and_ntiles)

#cummulative gains 
plot_cumgains(data = plot_input)

#Cumulative lift
plot_cumlift(data = plot_input)

#Response plot
plot_response(data = plot_input)

#Cumulative response plot
plot_cumresponse(data = plot_input)

#multiple plots
plot_multiplot(data = plot_input)


#Assessing Calibration

svm_prediction4a_ <-predict(svm_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(svm_prediction4a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#-----------------------------------------------------------------------------#
#6. Run the kNN Algorithm 
#Running over-sampling
set.seed(7777777)
knn_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "knn",
                        trControl = trControl1,metric= "F1")


#check for variable importance
knn_prediction1 <-predict(knn_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(knn_prediction1, test_set_ldd_1$ldd_bin)



#Parameter tuning
set.seed(7777777)
knn_ldd_model1a <- train(ldd_bin ~ ., 
                         data = train_set_ldd_1, method = "knn",
                         trControl = trControl1, metric= "F1",
                         tuneGrid = data.frame(k = seq(11,85,by = 2)))

knn_prediction1a <-predict(knn_ldd_model1a, test_set_ldd_1,type = "raw")
confusionMatrix(knn_prediction1a, test_set_ldd_1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
knn_data <- as.table(matrix(c(90,131,29,120), nrow = 2, byrow = TRUE)) #prediction Yes first col, pred NO second col
knn_rval <- epi.tests(knn_data, conf.level = 0.95,digits = 3)
print(knn_rval)

#F1 95%CI
f1_knn_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(knn_prediction1a) %>%
  rename(obs=ldd_bin, pred=knn_prediction1a) %>% mutate(obs=as.numeric(obs),
                                                        pred=as.numeric(pred))

bootout_knn1 <-boot(data=f1_knn_data1,
                    R=5000,
                    statistic=f1)

boot.ci(bootout_knn1,type="norm")
f1(f1_knn_data1)

#generating ROC Curve
roc_knn <- roc(test_set_ldd_1$ldd_bin,
               predict(knn_ldd_model1a, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
roc6<-ggroc(roc_knn, colour="Red") +
  ggtitle(paste0("KNN ROC Curve","(AUC=", round(pROC::auc(roc_knn),digits=4), ")")) +
  theme_minimal()

roc6

## Now plot
roc6a <- as.ggplot(function() plot(roc_knn, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc6a <- roc6a +
  labs(title=paste0("KNN ROC Curve","(AUC=", round(pROC::auc(roc_knn),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc6a

pROC::auc(roc_knn)
ci.auc(roc_knn)


#Calculating AUPRC
knn_prediction1a <-predict(knn_ldd_model1a, test_set_ldd_1,type ="prob")[,2]

knn_s1<- knn_prediction1a[1:74]
knn_s2 <- knn_prediction1a[75:148]
knn_s3 <- knn_prediction1a[149:222]
knn_s4 <- knn_prediction1a[223:296]
knn_s5 <- knn_prediction1a[297:370]

knn_s<- join_scores(knn_s1,knn_s2,knn_s3,knn_s4,knn_s5)
knn_mdat <- mmdata(scores=knn_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

knn_curves <- evalmod(scores=knn_prediction1a, labels= test_set_ldd_1$ldd_bin)
knn_curves1 <- evalmod(knn_mdat)

auc(knn_curves)
auc(knn_curves1)
auc_ci(knn_curves1,alpha = 0.05, dtype = "normal")

#Assessing Calibration

knn_prediction1a_ <-predict(knn_ldd_model1a, test_set_ldd_1,type = "prob")
val.prob(knn_prediction1a_$Yes, as.numeric(test_set_ldd_1_$ldd_bin))


#------------------------------------------------------------------------------#
#Run the nueralnet Algorithm 
#Running over-sampling
set.seed(3333333)
ann_ldd_model1 <- train(ldd_bin ~ ., 
                        data = train_set_ldd_1, method = "nnet",
                        trControl = trControl1,metric= "F1")

#check for variable importance
nn_prediction1 <-predict(ann_ldd_model1, test_set_ldd_1,type = "raw")
confusionMatrix(nn_prediction1, test_set_ldd_1$ldd_bin, mode='everything')


#computing 95% CI for performance metrics
nn_data <- as.table(matrix(c(98,142,21,109), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
nn_rval <- epi.tests(nn_data, conf.level = 0.95,digits = 3)
print(nn_rval)

f1_nn_data1 <- test_set_ldd_1 %>%
  dplyr::select(ldd_bin) %>% cbind(nn_prediction1) %>%
  rename(obs=ldd_bin, pred=nn_prediction1) %>% mutate(obs=as.numeric(obs),
                                                      pred=as.numeric(pred))

bootout_nn1 <-boot(data=f1_nn_data1,
                   R=5000,
                   statistic=f1)

boot.ci(bootout_nn1,type="norm")
f1(f1_nn_data1)

#generating ROC Curve
roc_nn <- roc(test_set_ldd_1$ldd_bin,
              predict(ann_ldd_model1, test_set_ldd_1, type = "prob")[,1],
              levels = rev(levels(test_set_ldd_1$ldd_bin)))

roc7<-ggroc(roc_nn, colour="Red") +
  ggtitle(paste0("ANN ROC Curve","(AUC=", round(pROC::auc(roc_nn),digits=4), ")")) +
  theme_minimal()

roc7

## Now plot
roc7a <- as.ggplot(function() plot(roc_nn, print.thres = c(.5), type = "S", 
                                   
                                   print.thres.cex = .8,
                                   legacy.axes = TRUE))

roc7a <- roc7a +
  labs(title=paste0("ANN ROC Curve","(AUC=", round(pROC::auc(roc_nn),digits=4), ")")) +
  theme(plot.title = element_text(hjust = 0.5))
roc7a

pROC::auc(roc_nn)
ci.auc(roc_nn)

#Calculating AUPRC
nn_prediction1 <-predict(ann_ldd_model1, test_set_ldd_1,type ="prob")[,2]

nn_s1<- nn_prediction1[1:74]
nn_s2 <- nn_prediction1[75:148]
nn_s3 <- nn_prediction1[149:222]
nn_s4 <- nn_prediction1[223:296]
nn_s5 <- nn_prediction1[297:370]

nn_s<- join_scores(nn_s1,nn_s2,nn_s3,nn_s4,nn_s5)
nn_mdat <- mmdata(scores=nn_s, labels= rf_l, modnames=c('m1'), dsids=1:5)

nn_curves <- evalmod(scores=nn_prediction1, labels= test_set_ldd_1$ldd_bin)
nn_curves1 <- evalmod(nn_mdat)

auc(nn_curves)
auc(nn_curves1)
auc_ci(nn_curves1,alpha = 0.05, dtype = "normal")

#computing shapley values using DALEX
explainer_ann <- explain( model = ann_ldd_model1, data = X, y = Y,label = "ANN",type = "classification")

explainer_ann1 <- update_data(explainer_ann, data = X1, y = Y1)


resids_ann <- model_performance(explainer_ann1)
plot(resids_ann)

mp_ann <- model_parts(explainer_ann1)
plot (mp_ann, show_boxplots = FALSE)

ann_shap <-predict_parts(explainer = explainer_ann,
                         new_observation = X1,
                         type = "shap",
                         B = 10 #number of reorderings - start small
)

ann_shap1 <- plot(ann_shap, show_boxplots = FALSE)
ann_shap1
#Assessing Calibration

nn_prediction1a_ <-predict(ann_ldd_model1, test_set_ldd_1,type = "prob")
val.prob(nn_prediction1a_$Yes, as.numeric(test_set_ldd_1$ldd_bin))


#Merging ROC Curves into one graph

grid.arrange(roc1,roc2,roc3,roc4,roc5,roc6,roc7, ncol = 3 )

grid.arrange(roc1a,roc2a,roc3a,roc4a,roc5a,roc6a,roc7a, ncol = 3)

grid.arrange(svm_shap1,ann_shap1,rf_shap1,gbm_shap1,  ncol=2)




##
##
##------------------------------------------------------------------------------------------------##
##--------------------------Stacked Ensemble using CaretEnsemble----------------------------------##

# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

control_stacking <- trainControl(method = "cv",
                                 number = 10,
                                 summaryFunction=f1,
                                 savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                                 classProbs = T, # To save the class probabilities of the out of fold predictions,
                                 sampling = "up")


algorithms_to_use <- c('rf','nnet',"lda","gam", "naive_bayes", "knn","xgbTree","svmLinear") # "xgbTree", 'knn', "adaboost"

ldd_stacked_models <- caretList(ldd_bin ~ ., data=train_set_ldd_1,
                                trControl=control_stacking, 
                                methodList=algorithms_to_use)

ldd_stacking_results <- resamples(ldd_stacked_models)

summary(ldd_stacking_results)

modelCor(resamples(ldd_stacked_models))

# stack using glm
stackControl <- trainControl(method = "cv",
                             number = 10,
                             savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                             classProbs = T,  # To save the class probabilities of the out of fold predictions,
                             sampling = "up")

fitGrid_2 <- expand.grid(mfinal = (1:3)*3,         
                         maxdepth = c(1, 3),      
                         coeflearn = c("Breiman"))

set.seed(77777777)
dur_glm_stack <- caretStack(ldd_stacked_models, method="glm", metric="Accuracy", trControl=stackControl)
dur_gbm_stack <- caretStack(ldd_stacked_models, method="gbm", metric="Accuracy", trControl=stackControl)

dur_glm_stack


model_preds <- lapply(ldd_stacked_models, predict, newdata=test_set_ldd_1)
model_preds <- lapply(model_preds, function(x) x[,"M"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(dur_glm_stack, newdata=test_set_ldd_1)
model_preds$ensemble <- ens_preds
confusionMatrix(ens_preds, test_set_ldd_1$ldd_bin, mode='everything')

ens_preds1 <- predict(dur_glm_stack, newdata=LDD_data_efgh1)
confusionMatrix(ens_preds1, LDD_data_efgh1$ldd_bin, mode='everything')

#computing 95% CI for performance metrics
ens_data <- as.table(matrix(c(88,61,31,190), nrow = 2, byrow = TRUE))#prediction Yes first col, pred NO second col
ens_rval <- epi.tests(ens_data, conf.level = 0.95,digits = 3)
print(ens_rval)


roc_ens <- roc(test_set_M$death,
               predict(dur_glm_stack, test_set_ldd_1, type = "prob")[,1],
               levels = rev(levels(test_set_ldd_1$ldd_bin)))
auc(roc_ens)

caTools::colAUC(model_preds, testing$Class)

save.image("C:/IEIP/DATA MANAGEMENT/Analysis/RSV prediction ML/Data/rsv_glm_stack.RData")

