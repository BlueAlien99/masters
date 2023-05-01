import random
import string


def get_random_string(n: int):
    char_pool = [*string.ascii_lowercase, *string.digits]
    return ''.join(random.SystemRandom().choice(char_pool) for _ in range(n))
