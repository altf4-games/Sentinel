import pandas as pd
import re

# Load the CSV file
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Function to clean individual elements
def clean_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Remove unwanted characters, including various quote types
        text = re.sub(r'[“”]', '', text)  # Remove smart quotes
        text = text.replace('"', '').replace("'", '').replace('\n', ' ')
        return ' '.join(text.split())  # Split and rejoin to normalize spaces
    return text  # Return as-is if not a string

# Apply the cleaning function to all columns in the DataFrame
df = df.applymap(clean_text)

# Remove empty rows
df.dropna(how='all', inplace=True)

# Remove rows with any empty columns
df.dropna(how='any', inplace=True)

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_file.csv'
df.to_csv(cleaned_file_path, index=False)

print(f'Preprocessing complete. Cleaned data saved to: {cleaned_file_path}')
