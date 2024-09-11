import yfinance as yf
import pandas as pd

# Define the stock symbol
symbol = 'AAPL'

# Fetch financial data from Yahoo Finance
stock = yf.Ticker(symbol)

# Income Statement
income_statement = stock.financials.T
print("Income Statement:")
print(income_statement.head())

# Balance Sheet
balance_sheet = stock.balance_sheet.T
print("\nBalance Sheet:")
print(balance_sheet.head())

# Cash Flow Statement
cashflow_statement = stock.cashflow.T
print("\nCash Flow Statement:")
print(cashflow_statement.head())

