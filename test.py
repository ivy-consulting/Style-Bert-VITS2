import string
def clean(sentence, dic):
    # Split the sentence into words
    words = sentence.split()
    # Remove punctuation from each word and keep only words that are in the dictionary (case-insensitive)
    filtered_words = [word for word in words if word.strip(string.punctuation).lower() in dic]
    # Join the filtered words back into a sentence
    return ' '.join(filtered_words)

dic = set()
dic.add("hello")
dic.add("world")
dic.add("thank")
dic.add("you")
dic.add("so")
dic.add("much")

print(clean("hello world! thank you so much.", dic))