import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text, include_stopwords=True):

    text = re.sub(r"\bcant\b", "can't", text)
    text = re.sub(r"\bwont\b", "won't", text)
    text = re.sub(r"\bshant\b", "shan't", text)
    text = re.sub(r"\bisnt\b", "isn't", text)
    text = re.sub(r"\barent\b", "aren't", text)
    text = re.sub(r"\bdont\b", "don't", text)
    text = re.sub(r"\bdoesnt\b", "doesn't", text)
    text = re.sub(r"\bdidnt\b", "didn't", text)
    text = re.sub(r"\bwasnt\b", "wasn't", text)
    text = re.sub(r"\bwerent\b", "weren't", text)
    text = re.sub(r"\bhasnt\b", "hasn't", text)
    text = re.sub(r"\bhavent\b", "haven't", text)
    text = re.sub(r"\bhadnt\b", "hadn't", text)
    text = re.sub(r"\bshouldnt\b", "shouldn't", text)
    text = re.sub(r"\bwouldnt\b", "wouldn't", text)
    text = re.sub(r"\bcouldnt\b", "couldn't", text)
    text = re.sub(r"\bmightnt\b", "mightn't", text)
    text = re.sub(r"\bneednt\b", "needn't", text)
    text = re.sub(r"\bmustnt\b", "mustn't", text)

    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bshan't\b", 'shall not', text)

    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\b(\w+)n't\b", r'\1 not', text)

    #text = re.sub(r"[^a-zA-Z0-9\s'!?]", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()

    if include_stopwords == False:

        #text = re.sub(r"[^a-zA-Z\s]", '', text)

        stop_words = set(stopwords.words('english'))
        stop_words.difference_update({"no", "not", "but", "very", "against", "until", "under", "again", "futher"})
        stop_words.update({"got", "would", "mc", "mcdonald", "mcdonalds", "staff", "service", "fast", "food", "order"})

        word_tokens = word_tokenize(text)

        text = [word for word in word_tokens if word.lower() not in stop_words]
        text = ' '.join(text)    

    return text