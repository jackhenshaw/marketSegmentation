import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv("data/customer_data.csv")

# change column names to lowercase
df.columns = map(str.lower, df.columns)

# drop customer_id as irrelevant for clustering people
df.drop('cust_id', axis=1, inplace=True)

# drop customer with no credit_limit (this was only 1 person)
df = df[df['credit_limit'].notna()]

# Find outliers in minimum_payments column
mp_cutoff = df['minimum_payments'].mean() + 5*df['minimum_payments'].std()
df['minimum_payments'] = df['minimum_payments'].apply(lambda mp: np.nan if mp > mp_cutoff else mp)
# Fill in missing values using iterative imputer
impute_it = IterativeImputer()
imputed = impute_it.fit_transform(df)
df = pd.DataFrame(imputed, columns=df.columns)

# Save new dataframe
df.to_csv("data/customer_data_complete.csv")