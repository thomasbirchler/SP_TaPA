def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_text_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)


def extract_text_after_sequence(file_path, sequence):
    extracted_lines = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line_index = 0

        for line in lines:
            if sequence in line:
                # Find the index of the sequence
                index = line.index(sequence)

                # Extract the text after the sequence
                extracted_text = line[index + len(sequence):].strip()

                # Append the extracted text to the result list
                extracted_lines.append(extracted_text)

                return extracted_lines, line_index
            line_index += 1


def read_lines_after_specified_line(file_path, specified_line):
    lines_after_specified_line = []
    record = False

    with open(file_path, 'r', encoding='utf-8') as file:
        line_number = 0
        for line in file:
            if specified_line is line_number:
                record = True
                line_number += 1
                continue  # Skip the specified line itself
            if record:
                lines_after_specified_line.append(line)
            line_number += 1

    string = '\n'.join(lines_after_specified_line)

    return string


def count_lines_in_file(file_path):
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_count += 1
    return line_count


def main():
    # Specify the paths to the two input text files
    for i in range(150, 210):
        for iteration in range(0, 4):
            # file1_path = f'test_{i}/prompt{iteration}.txt'
            file2_path = f'test_{i}/command{iteration}.txt'

            # Read the contents of both files
            # file1_content = read_text_file(file1_path)
            # file2_content = read_text_file(file2_path)

            # Specify the sequence to search for
            sequence = "my answer is:"
            # Find the additional text in file2
            first_line, index_answer_start = extract_text_after_sequence(file2_path, sequence)

            number_of_lines = count_lines_in_file(file2_path)
            if number_of_lines > index_answer_start:
                additional_lines = read_lines_after_specified_line(file2_path, index_answer_start)
                first_line.append(additional_lines)

            # Write the difference to a new text file
            output_file_path = f'test_{i}/command_only{iteration}.txt'
            my_string = '\n'.join(first_line)
            write_text_file(output_file_path, my_string)

            print(f"Difference written to '{output_file_path}'")


if __name__ == "__main__":
    main()
