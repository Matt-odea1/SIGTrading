import pandas as pd

def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

def exportToExcel(prices_df, output_file):
    with pd.ExcelWriter(output_file) as writer:
        prices_df.to_excel(writer, sheet_name='Prices', index=False)

# Load prices from the text file
prices_file = 'prices.txt'
prices_df = loadPrices(prices_file)

# Define output Excel file name
output_excel_file = 'newData.xlsx'

# Export to Excel
exportToExcel(prices_df, output_excel_file)

print(f"Data successfully exported to {output_excel_file}")