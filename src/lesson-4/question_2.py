"""
Use the following speech by the Rev. Dr. Martin Luther King, Jr.:

s = “I am happy to join with you today in what will go down in history as the greatest
demonstration for freedom in the history of our nation. Five score years ago,
a great American, in whose symbolic shadow we stand today, signed the Emancipation
Proclamation. This momentous decree came as a great beacon light of
hope to millions of Negro slaves who had been seared in the flames of withering injustice.
It came as a joyous daybreak to end the long night of their captivity. But one hundred years later,
the Negro still is not free. One hundred years later, the life of the Negro is still sadly
crippled by the manacles of segregation and the chains of discrimination.
One hundred years later, the Negro lives on a lonely island of poverty in the
midst of a vast ocean of material prosperity. One hundred years later, the Negro is
still languishing in the corners of American society and finds himself an exile in his
own land. So we have come here today to dramatize a shameful condition."

1. Find out how many unique words in s. (10 points)

2. Which word appears the most? (10 points)

3. How many words start with ‘t’. (10 points).
"""
import pandas as pd
import re

s = """
I am happy to join with you today in what will go down in history as the greatest
demonstration for freedom in the history of our nation. Five score years ago,
a great American, in whose symbolic shadow we stand today, signed the Emancipation
Proclamation. This momentous decree came as a great beacon light of
hope to millions of Negro slaves who had been seared in the flames of withering injustice.
It came as a joyous daybreak to end the long night of their captivity. But one hundred years later,
the Negro still is not free. One hundred years later, the life of the Negro is still sadly
crippled by the manacles of segregation and the chains of discrimination.
One hundred years later, the Negro lives on a lonely island of poverty in the
midst of a vast ocean of material prosperity. One hundred years later, the Negro is
still languishing in the corners of American society and finds himself an exile in his
own land. So we have come here today to dramatize a shameful condition.
"""

# 1. Find out how many unique words in s.
# extracting words using a regular expression (\b\w+\b)
# and converting them to lowercase.
# This process ignores punctuation and treats only alphanumeric sequences as words.
words = re.findall(r'\b\w+\b', s.lower())
word_counts = {}

# use a dictionary
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1

print("Number of unique words:", len(word_counts))

# use pandas
word_series = pd.Series(words)
word_count = word_series.value_counts()
print("Number of unique words:", len(word_count))


# 2. Which word appears the most?
most_common_word = word_count.idxmax()
most_common_count = word_count.max()

print("Most common word:", most_common_word)
print("Appears", most_common_count, "times")

# 3. How many words start with ‘t’.
t_words = [word for word in words if word.startswith('t')]
print("Number of words starting with 't':", len(t_words))

# using pandas
t_words_count = word_series.str.startswith('t').sum()
print("Number of words starting with 't':", t_words_count)


