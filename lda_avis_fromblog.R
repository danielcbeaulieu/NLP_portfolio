################## We need to call on our trusty friend PLYR ##########################
library(quanteda)
library(tm) # Framework for text mining.
library(qdap) # Quantitative discourse analysis of transcripts.
library(dplyr) # Data wrangling, pipe operator %>%().
library(RColorBrewer)# Generate palette of colours for plots.
library(ggplot2)# Plot word frequencies.
library(scales)# Include commas in numbers.
library(Rgraphviz)# Correlation plots.
library(plyr)
################## Set the working directory to the location of the text files #################
setwd("C:/Users/dabeaulieu/Documents/Python Scripts/text_analysis/cog/text")

################## Create a list of the files #####################################
file_list <- list.files()
#View(file_list)
################## Use the  lapply function to get the list ready#########################
filenames <- list.files("C:/Users/dabeaulieu/Documents/Python Scripts/text_analysis/cog/text", pattern="*.txt", full.names=TRUE)
filenames
mydata <- lapply(filenames, read.table, header = FALSE, sep="\t", quote=NULL, dec=".")
################## use the do.call function to 'row bind' each file in a data.frame##############
jsdf <- do.call(rbind, mydata)
#View(jsdf)
#str(jsdf)
#head(testcorp)
#summary(testcorp)
#View(testcorp$class)
testcorp <- Corpus(Dtestource("C:/Users/dabeaulieu/Documents/Python Scripts/text_analysis/cog/text")) #read in all text documents in directory
names(testcorp)
#testcorp2 <- tm_map(testcorp, content_transformer(tolower)) #convert to lower case
#toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) 
#testcorp2 <- tm_map(testcorp, toSpace, "/|@|\\|")
#testcorp2 <- tm_map(testcorp2, toSpace, "â")
#testcorp2 <- tm_map(testcorp2, toSpace, "???")
#testcorp2 <- tm_map(testcorp2, toSpace, "T")
#testcorp2 <- tm_map(testcorp2, toSpace, "-")
#testcorp2 <- tm_map(testcorp2, toSpace, "'")
#testcorp2 <- tm_map(testcorp2, toSpace, ".")
#testcorp2 <- tm_map(testcorp2, toSpace, "\"")

#testcorp <- tm_map(testcorp, toSpace, "-")
#testcorp <- tm_map(testcorp, toSpace, "'")
#testcorp <- tm_map(testcorp, toSpace, ".")
#testcorp <- tm_map(testcorp, toSpace, """)


#inspect(testcorp[1])
testcorp2 <- tm_map(testcorp, content_transformer(tolower)) #convert to lower case
#inspect(testcorp[14])
testcorppun <- tm_map(testcorp2, removeNumbers) #remove numbers
#inspect(testcorp[14])
testcorppun <- tm_map(testcorppun, removePunctuation) #remove punctuation
#inspect(testcorppun[14])
testcorppun <- tm_map(testcorppun, removeWords, stopwords("english")) #remove common words
#inspect(testcorppun[14])
length(stopwords("english")) #number of stopwords to be removed
stopwords("english") #show stopwords
testcorppun <- tm_map(testcorppun, removeWords, c("russian", "chinese", "testgov", "korean", "spanish")) #remove customs stopwords NOTE: need to find more words to remove
testcorppun <- tm_map(testcorppun, stripWhitespace) #Remove whitespace from text
#inspect(testcorppun[14])
myStopwords <- c("can", "say","one","way","use", "also","howev","tell","will","much","need","take","tend","even",
                 "like","particular","rather","said","get","well","make","ask","come","end",
                 "ftestt","two","help","often","may","might","see","someth","thing","point",
                 "post","look","right","now","think","'ve ","'re ")
#remove custom stopwords
testcorppun <- tm_map(testcorppun, removeWords, myStopwords)
testcorpfin <- tm_map(testcorppun, stemDocument) #Stemming uses an algorith to remove common word endings for English words, such as ES or ED or S
#inspect(testcorpfin[14])

#summary(testcorp)

llistopic.dtm <- tm::DocumentTermMatrix(testcorpfin, control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))
head.matrix(llistopic.dtm)
testdtm <- tm::DocumentTermMatrix(testcorpfin, control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))
head.matrix(testdtm)
term_tfidf <- tapply(llistopic.dtm$v/slam::row_sums(llistopic.dtm)[llistopic.dtm$i], llistopic.dtm$j, mean) *
  log2(tm::nDocs(llistopic.dtm)/slam::col_sums(llistopic.dtm > 0))
summary(term_tfidf)

## Keeping the rows with tfidf >= to the 0.155
llisreduced.dtm <- llistopic.dtm[,term_tfidf >= 0.0002993]
summary(slam::col_sums(llisreduced.dtm))

harmonicMean <- function(logLikelihoods, precision = 2000L) {
  llMed <- median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

k <- 17
burnin <- 1000
iter <- 1000 
keep <- 50
fitted <- topicmodels::LDA(llistopic.dtm, k = k, method = "Gibbs",control = list(burnin = burnin, iter = iter, keep = keep))
## assuming that burnin is a multiple of keep
logLiks <- fitted@logLiks[-c(1:(burnin/keep))]

## This returns the harmomnic mean for k = 25 topics.
library(Rmpfr)
harmonicMean(logLiks)

seqk <- seq(2, 100, 1)
burnin <- 1000
iter <- 1000
keep <- 50
system.time(fitted_many <- lapply(seqk, function(k) topicmodels::LDA(llisreduced.dtm, k = k,
                                                                     method = "Gibbs",control = list(burnin = burnin,
                                                                                                     iter = iter, keep = keep) )))
# extract logliks from each topic
logLiks_many <- lapply(fitted_many, function(L)  L@logLiks[-c(1:(burnin/keep))])
View(logLiks_many)
# compute harmonic means
hm_many <- sapply(logLiks_many, function(h) harmonicMean(h))

ldaplot <- ggplot(data.frame(seqk, hm_many), aes(x=seqk, y=hm_many)) + geom_path(lwd=1.5) +
  theme(text = element_text(family= NULL),
        axis.title.y=element_text(vjust=1, size=16),
        axis.title.x=element_text(vjust=-.5, size=16),
        axis.text=element_text(size=16),
        plot.title=element_text(size=20)) +
  xlab('Number of Topics') +
  ylab('Harmonic Mean') +
  annotate("text", x = 25, y = -80000, label = paste("The optimal number of topics is", seqk[which.max(hm_many)])) +
  ggtitle(expression(atop("Latent Dirichlet Allocation Analysis of NEN LLIS", atop(italic("How many distinct topics in the abstracts?"), ""))))

ldaplot

system.time(llis.model <- topicmodels::LDA(llisreduced.dtm, 19, method = "Gibbs", control = list(iter=2000, seed = 0622)))

str(llis.model)
llis.topics <- topicmodels::topics(llis.model, 19)
#llis.topics is the output for the topics by doucment

View(llis.topics) #USE THIS FOR MODEL
## In this case I am returning the top 80 terms because 30 is the optimal number of topics.
llis.terms <- as.data.frame(topicmodels::terms(llis.model, 66), stringsAsFactors = FALSE)
View(llis.terms) # this is teh second output we need 


