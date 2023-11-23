# Corruption Index and Business Open Days Clustering

This project utilizes K-means clustering to group countries based on their Corruption Index and the number of days it takes to open a business. The code uses the R programming language and several libraries, including `tidyverse`, `factoextra`, `stats`, `cluster`, and `gridExtra`, to perform the clustering and visualize the results.

## Requirements

Before running the code, ensure that you have the following R packages installed:

- `tidyverse`
- `factoextra`

You can install these packages using the following R commands:

```R
install.packages("tidyverse")
install.packages("factoextra")
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/parthdattani/ease-of-business-kmeans-clustering.git
   ```

2. Navigate to the repository directory:

   ```bash
   cd ease-of-business-kmeans-clustering
   ```

3. Run the R script `EaseOfBusinessRatingKmeansClustering.R` in your R environment or RStudio.

## Description

The code performs the following tasks:

1. Loads the required libraries, including `tidyverse`, `factoextra`, `stats`, `cluster`, and `gridExtra`.

2. Reads the dataset, "CountryData.csv," containing information about various countries, including their Corruption Index and Days to Open a Business.

3. Cleans the dataset by removing rows with missing data and converting the country name column to row names.

4. Scales the data to prepare it for clustering, selecting only the Corruption Index and Days to Open Business.

5. Generates clusters using K-means with an initial choice of 4 clusters and optimizes the number of clusters using the Elbow Method, Average Silhouette Method, and Gap Statistic.

6. Re-generates the cluster analysis with the optimal number of clusters (3).

7. Visualizes the clusters using `fviz_cluster` from the `factoextra` library.

8. Determines similarities and differences among the clusters using other features in the dataset and calculates their mean values.

## Dataset

The dataset, "CountryData.csv," contains information about various countries, including:

- Corruption Index
- Days to Open a Business
- Gini Coefficient
- GDP per Capita
- Education Percentage of Government Spending
- Education Percentage of GDP
- Compulsory Education Years

Please ensure you have this dataset in the same directory as the R script or update the file path accordingly.
