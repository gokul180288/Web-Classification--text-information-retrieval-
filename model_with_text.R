##### Code for StumbleUpon Classification Model - Kaggle (Text + Numerical variables) ########

library(tm)
library(SnowballC)
library(wordcloud)
library(jsonlite)


## Import data
training = read.table("train.tsv", sep = "\t", header = TRUE, stringsAsFactors = FALSE, na.strings=c("?"))

# Remove text columns and not useful column for handling numeric/factor features
data = training[,-c(1,3,13)]

############################## Handling none text part #######################################
#############################################################################################

# Na --> 
data$alchemy_category[is.na(data$alchemy_category)] = 'unknown'
data$is_news[is.na(data$is_news)] = 2
data$news_front_page[is.na(data$news_front_page)] = 5


data$alchemy_category_score = as.numeric(data$alchemy_category_score)
# change to factors
levels(data$alchemy_category) 
str(data$alchemy_category)   
data$is_news = as.factor(data$is_news)
data$news_front_page = as.factor(data$news_front_page)
data$hasDomainLink = as.factor(data$hasDomainLink)
data$lengthyLinkDomain = as.factor(data$lengthyLinkDomain)
data$hasDomainLink = as.factor(data$hasDomainLink)
data$alchemy_category = as.factor(data$alchemy_category)

data[is.na(data)] <- 999

data3<-data

library(dummies)
data3<-dummy.data.frame(data3,dummy.classes="factor")
data3$label<-as.factor(data3$label)


str(data3)




################################# Handling Text Part ###########################################
#################################################################################################


train.json = sapply(training$boilerplate, fromJSON)

################Corpus processing
#################################

review_corpus = Corpus(VectorSource(train.json))

# Change to lower case, not necessary here
review_corpus = tm_map(review_corpus, content_transformer(tolower))

# Remove numbers
review_corpus = tm_map(review_corpus, removeNumbers)

# Remove punctuation marks and stopwords
review_corpus = tm_map(review_corpus, removeWords, stopwords("english"))
review_corpus = tm_map(review_corpus, removePunctuation)

# Remove extra whitespaces
review_corpus =  tm_map(review_corpus, stripWhitespace)

# Convert to plain text
review_corpus <- tm_map(review_corpus, PlainTextDocument)

inspect(review_corpus[1])


# Document-Term Matrix: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Notice the dimension of the matrix
review_dtm <- DocumentTermMatrix(review_corpus)

# Simple word cloud
findFreqTerms(review_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(review_dtm1)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

# Remove the less frequent terms such that the sparsity is less than 0.95
review_dtm1 = removeSparseTerms(review_dtm, 0.95)

review_dtm2 = removeSparseTerms(review_dtm, 0.9)
# The first document
inspect(review_dtm[1,1:20])

# tf–idf(term frequency–inverse document frequency) instead of the frequencies of the term as entries, tf-idf measures the relative importance of a word to a document
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf1 = removeSparseTerms(review_dtm_tfidf, 0.95)
# The first document
inspect(review_dtm_tfidf[1,1:20])

# A new word cloud
freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))


##########LDA###############

library(lda)
set.seed(357)
fit <- lda.collapsed.gibbs.sampler(documents = review_dtm, K = 6, 
                                   num.iterations = 500, alpha = 0.05, 
                                   eta = 0.1, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)


### Topic modeling ###
library("topicmodels")

# Fitting a 10-topic model with variational EM
topics <- LDA(review_dtm, 10)
# Print the representative terms for each topic
terms(topics, 10)


################################## New data creation combining text and numeric features ##########
###################################################################################################

set.seed(8744231)
subset <- sample(nrow(data3), nrow(data3) * 0.8)
train = cbind(data3[subset, ],as.matrix(review_dtm_tfidf1)[subset,])
test = cbind(data3[-subset, ],as.matrix(review_dtm_tfidf1)[-subset,])

head(train)

# Now you know how to do the rest
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071) # for Support Vector Machine
library(nnet)

######### Model training

train_new= train[,2:560]

test_new = test[,2:560]

rf_model<-train(label~.,data=train_new,method="rf",
                trControl=trainControl(method="cv",number=4),
                prox=TRUE,allowParallel=TRUE)

plsClasses <- predict(rf_model, newdata = test_new)

confusionMatrix( test$label, data = plsClasses)

auc(test$label, plsClasses)

## fit cart model

head(train_new)

green.rpart <- rpart(formula = label ~ ., data = train_new)


plot(green.rpart, uniform = T, margin = 0.1, compress = T)
text(green.rpart, all=T)

rpart.plot(green.rpart, tweak = 0.8, extra=6,
           box.col=c("pink", "palegreen3")[green.rpart$frame$yval])

pred_outp1 <- predict(green.rpart, test_new, type = 'class')

table(test$label,pred_outp1, dnn = c("actual", "prediction"))

auc_cart =auc(test$label,pred_outp1)




