import os
import pandas as pd


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def main():
    # Directory containing text files
    directory = os.getcwd()

    # Initialize data as an empty list to store rows
    data = []

    # Loop through text files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("caption") and filename.endswith(".txt"):
            parts = filename.split('_')
            number = parts[0].replace('caption', '')
            direction = parts[1]
            prompt_number = parts[2].replace('.txt', '')

            # Read the content of the text file
            content = read_text_file(os.path.join(directory, filename))

            # Append the data as a dictionary to the list
            data.append({
                'Number': number,
                'Direction': direction,
                'Prompt Number': prompt_number,
                'Content': content
            })

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    excel_filename = 'captions.xlsx'  # Output Excel file name
    df.to_excel(excel_filename, index=False)

    print(f'Data has been written to {excel_filename}.')


if __name__ == "__main__":
    main()
