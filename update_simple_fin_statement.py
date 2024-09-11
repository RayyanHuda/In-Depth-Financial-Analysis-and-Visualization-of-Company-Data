import pandas as pd

# Load the new datasets
income_statement = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/income_statement.csv')
balance_sheet = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/balance_sheet.csv')
cash_flow = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/cash_flow.csv')

# Print the datasets
print("Income Statement:")
print(income_statement)

print("\nBalance Sheet:")
print(balance_sheet)

print("\nCash Flow Statement:")
print(cash_flow)

# Example analysis
# Calculate Net Profit Margin
income_statement['Net Profit Margin'] = income_statement['netIncome'] / income_statement['operatingIncome']
print("\nNet Profit Margin:")
print(income_statement[['date', 'Net Profit Margin']])

# Calculate Debt to Equity Ratio
balance_sheet['Debt to Equity Ratio'] = balance_sheet['totalLiabilities'] / balance_sheet['shareholdersEquity']
print("\nDebt to Equity Ratio:")
print(balance_sheet[['date', 'Debt to Equity Ratio']])

# Example: Plotting Net Income over Time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(income_statement['date'], income_statement['netIncome'], marker='o')
plt.xlabel('Date')
plt.ylabel('Net Income')
plt.title('Net Income Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()