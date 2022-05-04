
import regex as re


def clean_text(text):
    '''
    remove special characters, lower
    :param text:
    :return: cleaned text
    '''
    text = re.sub("[^A-Za-z0-9?!(),.'$%:-]+", " ", text)
    return text.lower().strip()

