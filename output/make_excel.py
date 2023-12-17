import pandas as pd


def count_left_occurrences(file_path, word):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Count the occurrences of "left" (case-insensitive)
            count = content.lower().count(word)
        return count
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return 0


def main():
    columns = ["TestNumber", "Temperature: 0.1", "Temperature: 0.35", "Temperature: 0.6", "top_k: 20", "top_k: 40", "top_k: 80", "top_k: 160", "max_tokens: 1", "max_tokens: 4", "max_tokens: 16", "max_tokens: 64", "max_tokens: 256"]
    df = pd.DataFrame(columns=columns)

    dtypes = {"TestNumber": int}
    for column in columns[1:]:
        dtypes[column] = float
    df = df.astype(dtypes)

    test_number = 150

    temp = 0.1
    top_k = 20
    max_tok = 1

    for i in range(0, 60):
        left_count = 0
        front_count = 0
        right_count = 0

        for iteration in range(0, 4):
            if iteration % 2 == 0:
                # Specify the path to the input text file
                file_path = f'test_{test_number}/command_only{iteration}.txt'

                # Count the occurrences of "left" in the file
                left_count += count_left_occurrences(file_path, "left")
                left_count += count_left_occurrences(file_path, "Left")
                left_count += count_left_occurrences(file_path, "LEFT")

                # Count the occurrences of right in the file
                right_count += count_left_occurrences(file_path, "right")
                right_count += count_left_occurrences(file_path, "Right")
                right_count += count_left_occurrences(file_path, "RIGHT")

                # Count the occurrences of front in the file
                front_count += count_left_occurrences(file_path, "front")
                front_count += count_left_occurrences(file_path, "Front")
                front_count += count_left_occurrences(file_path, "FRONT")

        denominator = left_count + front_count + right_count
        success_rate = 0
        if denominator != 0:
            success_rate = left_count / denominator

        df.loc[test_number, "TestNumber"] = test_number
        df.loc[test_number, f"Temperature: {temp}"] = success_rate
        df.loc[test_number, f"top_k: {top_k}"] = success_rate
        df.loc[test_number, f"max_tokens: {max_tok}"] = success_rate

        test_number += 1
        temp += 0.25
        if temp == 0.85:
            temp = 0.1
            top_k *= 2
            if top_k == 320:
                top_k = 20
                max_tok *= 4

    # Save the DataFrame as an Excel file
    excel_file_path = 'success_rate_without_captioning.xlsx'
    df.to_excel(excel_file_path)


if __name__ == "__main__":
    main()