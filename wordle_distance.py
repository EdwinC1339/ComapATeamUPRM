import numpy as np
from math import log2


def safe_log2(n):
    return log2(n) if n > 0 else 0


MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)


class WordleDistance:
    def __init__(self, words: np.ndarray[str]):
        self.words = words

    def wordle_distance(self, guess: str, target: str):
        return safe_log2(self.information(guess, target))

    def information(self, guess: str, target: str):
        # What's the information gained from guessing a word for a certain target?
        # First we must consider the pattern we will obtain
        pattern = WordleDistance.pattern(guess, target)

        # Then we will consider all words that will also produce that pattern
        pattern_matrix = np.array([
            WordleDistance.pattern(guess, w) for w in self.words
        ])


    @staticmethod
    def pattern(guess: str, target: str):
        guess_arr = np.array(list(guess))
        target_arr = np.array(list(target))
        pat = np.zeros([5], dtype=np.uint8)

        for i, c in enumerate(guess_arr):
            if target_arr[i] == c:
                pat[i] = EXACT
                # If we matched a character, we want to remove it from the target so we don't match it again.
                target_arr[i] = ''

        for i, c in enumerate(guess_arr):
            if c in target_arr and pat[i] != EXACT:
                pat[i] = MISPLACED
                x = np.where(target_arr == c)[0][0]
                target_arr[x] = ''

        return pat


print(WordleDistance.pattern('frame', 'eerie'))
