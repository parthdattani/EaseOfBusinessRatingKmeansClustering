# The following code clusters the corruption index and the number of days it
# requires to open a business using kmeans clustering

# Installing the tidyverse and factoextra packages
# install.packages("tidyverse")
install.packages("factoextra")

# Loading the tidyverse, stats, factoextra, cluster, and gridExtra libraries
library(tidyverse)
library(factoextra)
library(stats)
library(cluster)
library(gridExtra)

# Setting the working directory to your Lab11 folder
setwd("C:/Users/ual-laptop/Desktop/Eller Fall 2022/MIS545/Lab11")

# Reading CountryData.csv into an object called countries
countries <- read_csv("CountryData.csv",
                      col_names = TRUE,
                      col_types = "cnnnnini")

# Displaying the countries tibble on the console
print(countries)

# Displaying the structure of the countries tibble
str(countries)

# Displaying a summary of the countries tibble
summary(countries)

# Converting the column containing the country name to the 
# row title of the tibble
countries <- countries %>% column_to_rownames(var = "Country")

# Removing countries from the tibble with missing data in 
# any feature
countries <- countries %>% drop_na()
 
# Viewing the summary of the countries tibble again to 
# ensure there are no NA values
summary(countries)

# Scaling the tibble to take only CorruptionIndex and DaysToOpenBusiness
countriesScaled <- countries %>%
  select(CorruptionIndex, DaysToOpenBusiness) %>% scale()

# Setting the random seed to 679
set.seed(679)

# Generating the k-means clusters in an object called countries4Clusters 
# using 4 clusters and a value of 25 for nstart
countries4Clusters <- kmeans (x = countriesScaled,
                              centers = 4,
                              nstart = 25)

# Displaying cluster sizes on the console
countries4Clusters$size

# Displaying cluster centers (z-scores) on the console
countries4Clusters$centers

# Visualizing the clusters
fviz_cluster(object = countries4Clusters,
             data = countriesScaled,
             repel = FALSE)

# Optimizing the value for k by evaluating the elbow method
fviz_nbclust(x = countriesScaled,
             FUNcluster = kmeans,
             method = "wss")

# Optimizing the value for k by evaluating the Average Silhouette method
fviz_nbclust(x = countriesScaled,
             FUNcluster = kmeans,
             method = "silhouette")

# Optimizing the value for k by evaluating the Gap Statistic
fviz_nbclust(x = countriesScaled,
             FUNcluster = kmeans,
             method = "gap_stat")

# Regenerating the cluster analysis using the optimal number of clusters of 3
countries3Clusters <- kmeans (x = countriesScaled,
                              centers = 3,
                              nstart = 25)

# Displaying cluster sizes on the console
countries3Clusters$size

# Displaying cluster centers (z-scores) on the console
countries3Clusters$centers

# Visualizing the clusters
fviz_cluster(object = countries3Clusters,
             data = countriesScaled,
             repel = FALSE)

# Determining similarities and differences among the clusters using the 
# remaining features in the dataset
countries_clusters <- countries %>%
  mutate(cluster = countries3Clusters$cluster) %>%
  select(cluster,
         GiniCoefficient, 
         GDPPerCapita,
         EduPercGovSpend, 
         EduPercGDP,
         CompulsoryEducationYears
         ) %>%
  group_by(cluster) %>%
  summarize_all("mean")

print(countries_clusters)
