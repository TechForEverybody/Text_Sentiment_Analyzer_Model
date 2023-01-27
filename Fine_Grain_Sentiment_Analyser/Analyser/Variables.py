import re
from nltk.corpus import stopwords

Regular_expression_definition_for_html_tags=re.compile('<.*?>')
Regular_expression_definition_for_digits=re.compile('\d+\s|\s\d+|\s\d+\s')
Regular_expression_definition_for_links=re.compile('http://\S+|https://\S+')
english_stop_words=stopwords.words('english')

class_list=[
"admiration",
"anger",
"annoyance",
"approval",
"confusion",
"curiosity",
"desire",
"disappointment",
"disapproval",
"disgust",
"embarrassment",
"excitement",
"fear",
"gratitude",
"joy",
"love",
"nervousness",
"optimism",
"pride",
"realization",
"relief",
"remorse",
"sadness",
"surprise",
]

Sentiment_Level_Grouped_Emotions={
    "positive": [
        "amusement", 
        "excitement", 
        "joy", 
        "love", 
        "desire", 
        "optimism", 
        "caring", 
        "pride", 
        "admiration", 
        "gratitude", 
        "relief", 
        "approval"
    ],
    "negative": [
        "fear", 
        "nervousness", 
        "remorse", 
        "embarrassment", 
        "disappointment", 
        "sadness", 
        "grief", 
        "disgust", 
        "anger", 
        "annoyance", 
        "disapproval"
    ],
    "ambiguous": [
        "realization", 
        "surprise", 
        "curiosity", 
        "confusion"
    ]
}

Intermediate_Grouped_Emotions={
    "anger": [
        "anger", 
        "annoyance", 
        "disapproval"
    ],
    "disgust": ["disgust"],
    "fear": [
        "fear", 
        "nervousness"
    ],
    "joy": [
        "joy", 
        "amusement", 
        "approval", 
        "excitement", 
        "gratitude",  
        "love", 
        "optimism", 
        "relief", 
        "pride", 
        "admiration", 
        "desire", 
        "caring"
    ],
    "sadness": [
        "sadness", 
        "disappointment", 
        "embarrassment", 
        "grief",  
        "remorse"
    ],
    "surprise": [
        "surprise", 
        "realization", 
        "confusion", 
        "curiosity"
    ]
}

selected_class_list=[
"admiration",
"anger",
"annoyance",
"approval",
"confusion",
"curiosity",
"desire",
"disappointment",
"disapproval",
"disgust",
"embarrassment",
"excitement",
"fear",
"gratitude",
"joy",
"love",
"nervousness",
"optimism",
"pride",
"realization",
"relief",
"remorse",
"sadness",
"surprise",
]