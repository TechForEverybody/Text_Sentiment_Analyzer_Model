from .Variables import *
import pickle
from nltk.stem import WordNetLemmatizer
import joblib
from scipy.sparse import csr_matrix
import numpy

word_lemitizer = WordNetLemmatizer()


class TextInputProcessor:
    """
    TextInputProcessor: it includes the input text processor ans text vectorizer
    """
    def __init__(self):        
        print("""🚀🚀🚀Text input Processor🚀🚀🚀""")
    def setVecorizer(self,path:str)-> None:
        """
        Args:
            path (str): Path for the text Vectorizer
        """
        with open(path,"rb") as f:
            self.Vectorizer=pickle.load(f)
    def getPreprocessedInput(self,inputText:str) -> list:
        """
        Args:
            inputText (str): input sentence or paragraph

        Returns:
            list: list of preprocessed word of sentence
        """
        inputText=Regular_expression_definition_for_html_tags.sub(r" ",inputText)
        inputText=Regular_expression_definition_for_digits.sub(r" ",inputText)
        inputText=Regular_expression_definition_for_links.sub(r" ",inputText)
        punctuations = [".","=","_","<",">",",","!","?","'",'"',":",";","*","-","/","+","%","$","#","@","(",")","[","]","{","}",'\n']
        for i in punctuations:
            inputText = inputText.replace(i," ")
        inputText=inputText.lower().split()
        inputText=[word_lemitizer.lemmatize(word) for word in inputText]
        return inputText
    def getPreprocessTheArray(self,inputText:str)-> csr_matrix:
        """
        Args:
            inputText (str): input sentence or paragraph

        Returns:
            csr_matrix: Vector of given text
        """
        inputText=[ text for text in inputText.split(".") if len(text)>1]
        Processed_Text=[" ".join(self.getPreprocessedInput(sentence)) for sentence in inputText]
        print(Processed_Text)
        return self.Vectorizer.transform(Processed_Text)
    def getVectorArray(self,inputVector:csr_matrix)->list:
        """
        Args:
            inputVector (csr_matrix): Vector of given text

        Returns:
            list: one dimensional array
        """
        final_array=[]
        for j in [i for i in inputVector.toarray()]:
            for k in j:
                final_array.append(k)
        return final_array
    def getVectorizedForm(self,inputText:str)->csr_matrix:
        """
        Args:
            inputText (str): input sentence or paragraph

        Returns:
            csr_matrix: Vector of given text
        """
        return self.Vectorizer.transform([inputText])


class TextSentimentAnalyser:
    """TextSentimentAnalyser: It includes sentiment predictor in three different version of classes
    1. Traditional
    2. Fine-grained
    3. Defaults
    """
    def __init__(self):
        print("""🚀🚀🚀Text input Analyzer🚀🚀🚀""")
    def setModel(self,path:str)->None:    
        """
        Args:
            path (str): path for ML Model
        """
        self.model=joblib.load(path)
    def predict(self,inputVector:csr_matrix)->numpy.array:
        """
        Args:
            inputVector (csr_matrix): Vector of given text

        Returns:
            numpy.array: prediction
        """
        return self.model.predict(inputVector)[0]    
    def getCumulativePrediction(self,inputVector:csr_matrix)->str:
        """
        Args:
            inputVector (csr_matrix): Vector of given text

        Returns:
            str: max voted class
        """
        predictions=list(self.model.predict(inputVector))
        unique_value=list(set(predictions))
        max_voted_value=predictions[0]
        max_vote=predictions.count(max_voted_value)
        for i in unique_value:
            if max_vote<predictions.count(i):
                max_voted_value=i
                max_vote=predictions.count(i)
        return max_voted_value
    def getInterMediateCumulativePrediction(self,inputVector:csr_matrix)->str:
        """
        Args:
            inputVector (csr_matrix): Vector of given text

        Returns:
            str: max voted intermediate class
        """
        predictions=list(self.getArrayofIntermediateEmotions(self.model.predict(inputVector)))
        unique_value=list(set(predictions))
        max_voted_value=predictions[0]
        max_vote=predictions.count(max_voted_value)
        for i in unique_value:
            if max_vote<predictions.count(i):
                max_voted_value=i
                max_vote=predictions.count(i)
        return max_voted_value
    def getTraditionalCumulativePrediction(self,inputVector:csr_matrix)->str:
        """
        Args:
            inputVector (csr_matrix): Vector of given text

        Returns:
            str: max voted traditional class
        """
        predictions=list(self.getArrayofSentimentLevelEmotions(self.model.predict(inputVector)))
        unique_value=list(set(predictions))
        max_voted_value=predictions[0]
        max_vote=predictions.count(max_voted_value)
        for i in unique_value:
            if max_vote<predictions.count(i):
                max_voted_value=i
                max_vote=predictions.count(i)
        return max_voted_value
    def getIntermediateEmotions(self,result:numpy.array)->str:
        """
        Args:
            result (numpy.array): _description_

        Returns:
            str: intermediate class
        """
        for key in Intermediate_Grouped_Emotions.keys():
            if result in Intermediate_Grouped_Emotions[key]:
                return key
        return "Neutral"
    def getSentimentLevelEmotions(self,result:numpy.array)->str:
        """
        Args:
            result (numpy.array): _description_

        Returns:
            str: Traditional class
        """
        for key in Sentiment_Level_Grouped_Emotions.keys():
            if result in Sentiment_Level_Grouped_Emotions[key]:
                return key
        return "Neutral"
    def getArrayofIntermediateEmotions(self,result:numpy.array)->list:
        """
        Args:
            result (numpy.array): _description_

        Returns:
            list: list of intermediate classes
        """
        output=[]
        for value in result:
            for key in Intermediate_Grouped_Emotions.keys():
                if value in Intermediate_Grouped_Emotions[key]:
                    output.append(key)
        return output
    def getArrayofSentimentLevelEmotions(self,result:numpy.array)->list:
        """
        Args:
            result (numpy.array): _description_

        Returns:
            list: list of traditional classes
        """
        output=[]
        for value in result:
            for key in Sentiment_Level_Grouped_Emotions.keys():
                if value in Sentiment_Level_Grouped_Emotions[key]:
                    output.append(key)
        return output
