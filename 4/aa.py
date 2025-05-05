import pandas as pd
import random

# Reduced size to avoid memory issue
def generate_customer_data(num_rows, churn_value, start_id):
    data = []
    for i in range(num_rows):
        customer_id = start_id + i
        gender = random.choice(["Male", "Female"])
        age = random.randint(18, 65)
        tenure = random.randint(0, 10)
        balance = round(random.uniform(0, 250000), 2)
        num_of_products = random.randint(1, 4)
        has_cr_card = random.randint(0, 1)
        is_active_member = random.randint(0, 1)
        estimated_salary = round(random.uniform(10000, 200000), 2)
        data.append([
            customer_id, gender, age, tenure, balance,
            num_of_products, has_cr_card, is_active_member,
            estimated_salary, churn_value
        ])
    return data

# Generate smaller dataset
churn_1_data = generate_customer_data(100, 1, 30000000)
churn_0_data = generate_customer_data(100, 0, 30100000)

# Combine data and create DataFrame
columns = [
    "CustomerID", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Churn"
]
df_small = pd.DataFrame(churn_1_data + churn_0_data, columns=columns)

# Save to CSV
file_path_small = "small_synthetic_churn_data.csv"
df_small.to_csv(file_path_small, index=False)

file_path_small
