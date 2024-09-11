import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load datasets
try:
    income_statement = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/income_statement.csv')
    balance_sheet = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/balance_sheet.csv')
    cash_flow = pd.read_csv('C:/Users/Rayya/Documents/UWO Folder BMOS/Certifcates.Resume.UniDocs/Projects/cash_flow.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Display datasets
print("\nIncome Statement:")
print(income_statement.head())

print("\nBalance Sheet:")
print(balance_sheet.head())

print("\nCash Flow Statement:")
print(cash_flow.head())

# Calculate financial ratios
if 'grossProfit' in income_statement.columns and 'totalRevenue' in income_statement.columns:
    income_statement['Gross Profit Margin'] = (income_statement['totalRevenue'] - income_statement['costOfRevenue']) / income_statement['totalRevenue']
if 'netIncome' in income_statement.columns and 'operatingIncome' in income_statement.columns:
    income_statement['Net Profit Margin'] = income_statement['netIncome'] / income_statement['operatingIncome']
if 'totalLiabilities' in balance_sheet.columns and 'shareholdersEquity' in balance_sheet.columns:
    balance_sheet['Debt to Equity Ratio'] = balance_sheet['totalLiabilities'] / balance_sheet['shareholdersEquity']
if 'Current Assets' in balance_sheet.columns and 'Inventory' in balance_sheet.columns and 'Current Liabilities' in balance_sheet.columns:
    balance_sheet['Quick Ratio'] = (balance_sheet['Current Assets'] - balance_sheet['Inventory']) / balance_sheet['Current Liabilities']
if 'netIncome' in income_statement.columns and 'totalAssets' in balance_sheet.columns:
    balance_sheet['ROA'] = income_statement['netIncome'] / balance_sheet['totalAssets']
if 'netIncome' in income_statement.columns and 'shareholdersEquity' in balance_sheet.columns:
    balance_sheet['ROE'] = income_statement['netIncome'] / balance_sheet['shareholdersEquity']
if 'totalRevenue' in income_statement.columns and 'totalAssets' in balance_sheet.columns:
    balance_sheet['Asset Turnover'] = income_statement['totalRevenue'] / balance_sheet['totalAssets']
if 'costOfRevenue' in income_statement.columns and 'Inventory' in balance_sheet.columns:
    balance_sheet['Inventory Turnover'] = income_statement['costOfRevenue'] / balance_sheet['Inventory']
if 'netIncome' in income_statement.columns and 'sharesOutstanding' in income_statement.columns:
    income_statement['EPS'] = income_statement['netIncome'] / income_statement['sharesOutstanding']

# Display ratios
print("\nFinancial Ratios:")
print(income_statement[['date', 'Gross Profit Margin', 'Net Profit Margin', 'EPS']].head())
print(balance_sheet[['date', 'Debt to Equity Ratio', 'Quick Ratio', 'ROA', 'ROE', 'Asset Turnover', 'Inventory Turnover']].head())

# Expense Breakdown
expense_breakdown = income_statement[['costOfRevenue', 'operatingExpense', 'interestExpense']].sum()
expense_breakdown.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Expense Breakdown')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Data Preparation for Linear Regression
X = income_statement[['totalRevenue', 'operatingExpense', 'costOfRevenue']]
y = income_statement['netIncome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\nMean Squared Error of the prediction: {mse}")

# Future Predictions
future_dates = pd.date_range(start=income_statement['date'].max(), periods=5, freq='Y')
future_X = X.tail(1).copy().reset_index(drop=True)
future_X = pd.concat([future_X] * len(future_dates), ignore_index=True)
future_X.index = future_dates
future_values = model.predict(future_X)
future_values = pd.DataFrame(future_values, index=future_dates, columns=['Predicted Net Income'])

print("\nFuture Predictions:")
print(future_values)

# Performance Metrics Summary
performance_summary = {
    'Metric': ['Net Profit Margin', 'Gross Profit Margin', 'ROA', 'ROE', 'Debt to Equity', 'Quick Ratio', 'EPS'],
    'Value': [
        income_statement['Net Profit Margin'].mean(),
        income_statement['Gross Profit Margin'].mean(),
        balance_sheet['ROA'].mean(),
        balance_sheet['ROE'].mean(),
        balance_sheet['Debt to Equity Ratio'].mean(),
        balance_sheet['Quick Ratio'].mean(),
        income_statement['EPS'].mean()
    ]
}

print("\nCompany Performance Summary:")
print(pd.DataFrame(performance_summary))

# Correlation Matrix
correlation_matrix = balance_sheet[['Debt to Equity Ratio', 'Net Profit Margin']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Between Debt to Equity and Net Profit Margin')
plt.show()

# Data visualization
plt.figure(figsize=(15, 12))

# Plot Net Income Over Time
if 'date' in income_statement.columns and 'netIncome' in income_statement.columns:
    plt.subplot(3, 1, 1)
    plt.plot(income_statement['date'], income_statement['netIncome'], marker='o', color='green', label='Net Income')
    plt.xlabel('Date')
    plt.ylabel('Net Income')
    plt.title('Net Income Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

# Plot ROE Over Time
if 'date' in balance_sheet.columns and 'ROE' in balance_sheet.columns:
    plt.subplot(3, 1, 2)
    plt.plot(balance_sheet['date'], balance_sheet['ROE'], marker='o', color='purple', label='ROE')
    plt.xlabel('Date')
    plt.ylabel('ROE')
    plt.title('Return on Equity (ROE) Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

# Plot Financial Performance Over Time
if 'date' in income_statement.columns and all(col in income_statement.columns for col in ['totalRevenue', 'netIncome', 'EPS']):
    plt.figure(figsize=(10, 6))
    plt.plot(income_statement['date'], income_statement['totalRevenue'], label='Revenue', marker='o')
    plt.plot(income_statement['date'], income_statement['netIncome'], label='Net Income', marker='o')
    plt.plot(income_statement['date'], income_statement['EPS'], label='EPS', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Financial Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Adjust layout for better visibility
plt.tight_layout()
plt.show()