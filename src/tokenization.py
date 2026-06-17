

def vocabularize(tokens):
    """Create a vocabulary from the list of tokens."""
    vocab = set(tokens)
    return list(vocab)


def char_tokenize(text):
    """Tokenize text at the character level."""
    tokens = list(text)
    vocab = vocabularize(tokens)
    # create a dict which converts characters to token ids
    char_to_id = {char: idx for idx, char in enumerate(vocab)}
    id_to_char = {idx: char for char, idx in char_to_id.items()}
    # create the token ids list
    token_ids = [char_to_id[char] for char in tokens]

    return token_ids, vocab, char_to_id, id_to_char



if __name__ == "__main__":
    input_file = "data/processed/hp1.txt"
    # load the cleaned data
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()
    # tokenize the text
    tokens = char_tokenize(text)
    # save the tokens to the output file
    print("stop")