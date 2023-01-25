# imports 
from Fine_Grain_Sentiment_Analyser import TextInputProcessor,TextSentimentAnalyser

# object creation
textInputProcessor=TextInputProcessor()
textSentimentAnalyser=TextSentimentAnalyser()

# Setting the Model Paths
textSentimentAnalyser.setModel("./Fine_Grain_Sentiment_Analyser/Model/Model.joblib")
textInputProcessor.setVecorizer("./Fine_Grain_Sentiment_Analyser/Model/Vectorizer.pickle")

# Raw Text Input
text_input="I need to try this Model"

# Preprocessing the Input
preprocessed_input=textInputProcessor.getPreprocessTheArray("not good")

# generating the result
sentiment_result=textSentimentAnalyser.predict(preprocessed_input)

print(sentiment_result)