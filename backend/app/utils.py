import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_word_freq(words, text):
    total_words = len(text.split())
    text = text.lower()

    word_freq = [(text.count(word) / total_words) * 100 for word in words]
    return word_freq


def get_character_freq(characters, text):
    total_characters = len(text)
    text = text.lower()

    character_freq = [(text.count(character) / total_characters) * 100 for character in characters]
    return character_freq


def get_capital_sequences(text):
    capital_sequences = []
    for word in text.split():
        if word.isupper():
            capital_sequences.append(word)
    return capital_sequences


def get_word_lengths(words):
    return [len(word) for word in words]


def get_email_features(text, words, characters):
    features = []
    features.extend(get_word_freq(words, text))
    features.extend(get_character_freq(characters, text))

    capital_sequences = get_capital_sequences(text)
    capital_sequences_lengths = get_word_lengths(capital_sequences)
    if not capital_sequences_lengths:
        features.extend([0, 0, 0])
    else:
        features.extend(
            [np.mean(capital_sequences_lengths), np.max(capital_sequences_lengths), np.sum(capital_sequences_lengths)])

    return np.array(features)


def split_feature_names(feature_names):
    words = [name[10:] for name in feature_names if name.startswith('word_freq')]
    characters = [name[10:] for name in feature_names if name.startswith('char_freq')]

    return words, characters

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return generated_text
