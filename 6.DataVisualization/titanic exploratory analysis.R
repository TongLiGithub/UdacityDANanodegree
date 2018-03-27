library(Amelia)
library(psych)


titanic<-read.csv('titanic-data.csv')
head(titanic)
summary(titanic)

titanic$Survival[titanic$Survived==0]="Perished"
titanic$Survival[titanic$Survived==1]="Survived"

sum(titanic$Survived)
titanic['count']=1

titanic$Age_group[titanic$Age==NA]=NA
titanic$Age_group[titanic$Age<=20]="0-20"
titanic$Age_group[titanic$Age<=40 & titanic$Age>20]="20-40"
titanic$Age_group[titanic$Age<=60 & titanic$Age>40]="40-60"
titanic$Age_group[titanic$Age>60]="60+"


write.csv(titanic, file='titanic.csv')

str(titanic)

#check for missing value
##1. Exclude PassengerId, as well as three factor variables Name, Ticket, Cabin. 
##   These three factor variables have too many levels.
subdata <- titanic[, c(2,3,5,6,7,8,10,12,13,14,15)]
str(subdata)
##2. transform factors to characters, because factors do not show missing values
titanic$Sex <- as.character(titanic$Sex)
titanic$Embarked <- as.character(titanic$Embarked)

sapply(subdata,function(x) sum(is.na(x)))
missmap(subdata, main = "Missing values vs observed")

#check for unique data
sapply(subdata,function(x) length(unique(x)))

#logistic regression 
model1 <- glm(Survived ~ .-Age-count-Survival, family = binomial(link = "logit"), 
              data = subdata)
summary(model1)

##according to model1, Parch, Fare, and Embarked did not contribute to the model.
##Therefore in model2 they were excluded.

model2 <- update(model1, .~.-Parch-Fare-Embarked)
summary(model2)

anova(model1, model2, test="Chisq")

##model2 used fewer variables but not different from model1, therefore keep using model2

model3 <- glm(Survived ~ Pclass*Sex+SibSp+Age_group, family = binomial(link = "logit"), 
              data = subdata)
summary(model3)

model4 <- glm(Survived ~ Pclass*Sex*Age_group+SibSp, family = binomial(link = "logit"), 
              data = subdata)
summary(model4)


anova(model2, model3, model4, test="Chisq")
#model3 is better than model2, but model3 and model4 are not different.
#Therefore use model3 as the final model.

#recode SibSp to 0 and 1, 0 for no siblings or spouse with the passenger, 
#and 1 for with siblings or spouse
subdata$SibSp_category[subdata$SibSp==0]<-0
subdata$SibSp_category[subdata$SibSp==1]<-1
subdata$SibSp_category[subdata$SibSp>1]<-2
subdata$SibSp_category <- factor(subdata$SibSp_category)

str(subdata$SibSp_category)

##try in the model
model5 <- glm(Survived ~ Pclass*Sex+SibSp_category+Age_group, family = binomial(link = "logit"), 
                   data = subdata)
summary(model5)

model6 <- glm(Survived ~ Pclass*Sex+SibSp_category*Sex+Age_group, family = binomial(link = "logit"), 
              data = subdata)
summary(model6)


titanic$SibSp_category[titanic$SibSp==0]<-"0"
titanic$SibSp_category[titanic$SibSp==1]<-"1"
titanic$SibSp_category[titanic$SibSp>1]<-"more"
titanic$SibSp_category <- as.character(titanic$SibSp_category)

write.csv(titanic, file='titanic2.csv')
