# Market Segmentation - An Exploration of Unsupervised Learning

**Context:** This project represents an early exploration into unsupervised learning and clustering techniques. It highlights my foundational understanding of these algorithms and, crucially, the challenges and considerations when applying them to real-world datasets lacking clear ground truth. Looking back with more industry experience, this project reinforced key lessons in feature engineering and robust model evaluation for business impact.

## Project Objective
The primary goal of this project was to develop customer segments from credit card usage data. This segmentation aims to enable targeted recommendations (e.g., saving plans, loan offers, wealth management products) to specific customer groups, optimising marketing efforts and customer engagement for a financial institution.

**Dataset:** The analysis utilises a public dataset summarising the 6-month usage behaviour of approximately 9000 active credit cardholders, featuring 18 behavioural variables. [Link to Kaggle Dataset](https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised)

## Methodology
This project followed a standard data science workflow:

1.  **Data Preprocessing:**
    * Conducted initial exploratory data analysis (EDA) to understand distributions and identify anomalies.
    * Addressed missing values using `IterativeImputer` for robust imputation.
    * Performed data validation and explored correlations between behavioural variables.
2.  **Clustering Model Experimentation:**
    * Explored and trained several unsupervised clustering algorithms, including K-Means, Mean Shift, Affinity Propagation, and Spectral Clustering, to identify natural groupings within the customer data.
    * Evaluated model performance primarily using the Silhouette Score, a common metric for assessing the compactness and separation of clusters.

## Key Findings & Reflection

* **Model Selection & Challenges:** While K-Means yielded the highest Silhouette Score (with 2 clusters), this limited separation was deemed insufficient for meaningful business segmentation. Mean Shift clustering, despite a lower Silhouette Score, produced 17 clusters, which conceptually offered a more granular view for potential targeted interventions. This highlights the critical balance between statistical metrics and real-world business utility in unsupervised learning.
* **Limitations & Insights on Unsupervised Learning:** The Silhouette Scores across all models remained below 0.3, indicating weak clustering performance and visually confirmed by PCA analysis where clusters lacked clear separation. This project underscored several key learnings:
    * **The Importance of Feature Engineering:** Robust feature engineering, potentially incorporating domain expertise, is crucial for improving cluster separability and model performance in unsupervised tasks.
    * **Beyond Silhouette:** Relying solely on internal metrics like Silhouette Score can be insufficient. Exploring external validation metrics (if ground truth were available) or business-driven evaluation metrics is essential.
    * **Business Context is King:** My current work on social tariff eligibility, where clusters are evaluated against a clear business outcome (eligibility for a social tariff), has further reinforced that the *interpretability* and *actionability* of clusters are often more important than a high internal statistical score. For instance, in the social tariff project, each cluster is analysed for the proportion of eligible customers, leading to clear business actions (marking groups eligible/not eligible). This market segmentation project, by contrast, highlighted the challenge of defining "success" without such explicit business-defined targets.

* **Code Structure:** The repository includes well-documented Python scripts for data cleaning, preprocessing, and the implementation of the chosen Mean Shift clustering model.
