sentence_buffer = []

def add_word(word):
    if len(sentence_buffer) == 0 or word != sentence_buffer[-1]:
        sentence_buffer.append(word)

def clear_sentence():
    sentence_buffer.clear()

def get_sentence():
    return " ".join(sentence_buffer)
