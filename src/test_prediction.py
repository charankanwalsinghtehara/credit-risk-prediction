
import pandas as pd

from predict import predict_credit_risk


df = pd.read_csv(
    "data/raw/application_train.csv"
)

customer = df.drop(
    columns=["TARGET"]
).iloc[0].to_dict()


result = predict_credit_risk(
    customer
)

print(result)