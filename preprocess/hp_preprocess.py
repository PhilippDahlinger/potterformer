

def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # Check if the line is not empty
                outfile.write(line)

if __name__ == "__main__":
    input_file = '../data/raw/hp1.txt'  # Replace with your actual input file path
    output_file = '../data/processed/hp1.txt'  # Replace with your desired output file path
    remove_empty_lines(input_file, output_file)
    print(f"Empty lines removed. Cleaned data saved to {output_file}.")