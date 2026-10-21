# PM-09: Set Up the Shared HR Dataset
import pandas as pd 
import numpy as np
np.random.seed(42) # Reproducible results n = 500 # 500 employee
departments = ['Sales', 'Support', 'Management', 'Engineering', 'Finance'] 
department_ids = {d: i for i, d in enumerate(departments)} 
n = 1000
department = np.random.choice(departments, n) 
years_of_service = np.random.randint(0, 25, n) 
is_manager = np.random.choice([0, 1], n, p=[0.85, 0.15]) 
performance_score = np.random.normal(3.5, 0.8, n).clip(1, 5).round(1) 
age = np.random.randint(22, 65, n) 
base = 15000 
dept_bonus = { 
    'Sales': 2000, 'Support': 0, 'Management': 15000, 
    'Engineering': 8000, 'Finance': 6000 
} 
salary = ( 
    base 
    + np.array([dept_bonus[d] for d in department]) 
    + years_of_service * 1200 
    + is_manager * 20000 
    + performance_score * 3000 
    + np.random.normal(0, 2000, n)  # noise 
) 
# Junior: < 35k, Mid: 35k-55k, Senior: > 55k 
salary_band = np.where(salary < 35000, 'Junior', 
              np.where(salary < 55000, 'Mid', 'Senior')) 
 
df = pd.DataFrame({ 
    'EmployeeID': range(1, n + 1), 
    'Department': department, 
    'DepartmentID': [department_ids[d] for d in department], 
    'YearsOfService': years_of_service, 
    'IsManager': is_manager, 
    'PerformanceScore': performance_score, 
    'Age': age, 
    'Salary': salary.round(2), 
    'SalaryBand': salary_band 
}) 

df.to_csv('hr_data.csv', index=False)
print(f"hr_data.csv created with {len(df)} rows") 
print("\nFirst 5 rows:") 
print(df.head()) 
print("\nSalary band distribution:") 
print(df['SalaryBand'].value_counts()) 
print(f"\nDataset shape: {df.shape}")
