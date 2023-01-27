# imports 
from Fine_Grain_Sentiment_Analyser import TextInputProcessor,TextSentimentAnalyser

# object creation
textInputProcessor:object=TextInputProcessor()
textSentimentAnalyser:object=TextSentimentAnalyser()

# Setting the Model Paths
textSentimentAnalyser.setModel("./Fine_Grain_Sentiment_Analyser/Model/Model.joblib")
textInputProcessor.setVecorizer("./Fine_Grain_Sentiment_Analyser/Model/Vectorizer.pickle")

# Raw Text Input
text_input:str="I need to try this Model"

# Preprocessing the Input
preprocessed_input=textInputProcessor.getPreprocessTheArray("good")

# generating the result
sentiment_result=textSentimentAnalyser.predict(preprocessed_input)

print(sentiment_result)