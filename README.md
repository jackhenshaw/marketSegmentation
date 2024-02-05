# Market Segmentation
## Objective
This case requires developing a customer segmentation to give recommendations like saving plans, loans, wealth management, etc. on target customer groups.

The sample Dataset summarizes the usage behavior of about 9000 active credit cardholders during the last 6 months. The file is at a customer level with 18 behavioral variables. <br>
Data obtained from: https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised

## Status
- Performed initial explore of data and filled in missing values using IterativeImputer.
- Checked values made sense and looked at correlations between variables
- Trained a few ML models and tested clustering with sillhouette score.
  - K-means, mean shift, affinity propagation, spectral clustering.
 
## Results
Best sillhouette score came from K-means with 2 clusters but based on the use case I made the decision that this was not enough separation for this clustering. <br>
Second best score came from mean shift clustering with 17 clusters and this was the model used moving forwards. 

The sillhouette scores of all the models was less than 0.3 which is low and showing only weak clustering performance. This was seen in the PCA analysis where clusters were not seperable to the human eye. Further work on feature engineering better separation would likely be useful. Additionally further checks of other metrics that also analyse the efficiency of clustering algorithms might be of use in separating the best model for the task.

Python code to first "clean" the data and then train the mean shift model have been written.
