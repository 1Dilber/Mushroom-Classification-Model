library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)

raw <- fread('mushrooms.csv')
df <- raw

df %>% names()
names(df) <- names(df) %>% str_replace_all('-','_') %>% 
  str_replace_all('%','_')

#Checking data
df %>% skim()

df$class <- df$class %>% as.factor()

df$class <- factor(df$class, levels = c("'e'", "'p'"), labels = c(0, 1)) 
class(df$class)

#check the proportion
df$class %>% table %>% prop.table

#split data
h2o.init()

h2o_data <- df %>% as.h2o()
h2o_data <- h2o_data %>% h2o.splitFrame(ratio=0.8, seed=123)


train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- df %>% select(-class) %>% names()

# 1. Build classification model with h2o.automl();
model <- h2o.automl(x=features,y=target,
                    training_frame=train,
                    validation_frame =test,
                    leaderboard_frame=test,
                    nfolds= 10,
                    max_runtime_secs = 360,
                    stopping_metric='AUC',
                    seed = 123)

model@leaderboard %>% as.data.frame()
leader <- model@leader

pred <- leader %>% h2o.predict(test)

# 2. Apply Cross-validation;

leader %>% h2o.confusionMatrix(test) %>% 
  as.tibble() %>% 
  select('0','1') %>% 
  .[1:2,] %>% t()%>% 
  fourfoldplot(conf.level=0,,color=c('red','darkgreen'),   
               main=paste('Accuracy=',
                          round(sum(diag(.))/sum(.)*100,1),'%'))

# 3. Find threshold by max F1 score;
threshold <- leader %>% h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1')

# 4. Calculate Accuracy, AUC, GİNİ.
leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)


# Complete Homework with following steps:
# 1. Name your final homework Script as ('Mushroom Classification Model')
# 2. Create repository named “Mushroom-Classification-Model” in your Github account and push your homework Script to this repository.
# 3. Fork other users’ repositories, make pull requests (at least one, making three pull requests is desirable).