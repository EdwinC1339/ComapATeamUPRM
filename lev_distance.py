from functools import lru_cache

def lev_distance(a: str, b: str) -> int:
    """
    This function calculates the Levenshtein distance between two input
    strings a and b.
    
    Args:
        a (str): The first string to compare.
        b (str): The second string to compare.
        
    Returns:
        The Levenshtein distance between string a and string b.
    """
    
    @lru_cache(None)
    def min_distance(s1: int, s2: int) -> int:
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        if a[s1] == b[s2]:
            return min_distance(s1 + 1, s2 + 1)

        return 1 + min(
            min_distance(s1, s2 + 1),      # insert character
            min_distance(s1 + 1, s2),      # delete character
            min_distance(s1 + 1, s2 + 1)   # replace character
        )

    return min_distance(0, 0)