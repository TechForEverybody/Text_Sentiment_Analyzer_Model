<h1>Text Sentiment Analyzer Model</h1>
<p>A Multi-class Classification for Many different Sentiments</p>
<h3>Created By: Shivkumar Chauhan</h3>


<hr>
<h1>Model Folder Structure</h1>

<pre>
📦Fine_Grain_Sentiment_Analyser
┣--------📂Analyser
┃--------┣--------📜TextProcessor.py
┃--------┣--------📜Variables.py
┃--------┗--------📜__init__.py
┣--------📂Model
┃--------┣--------📜Model.joblib
┃--------┗--------📜Vectorizer.pickle
┗--------📜__init__.py
</pre>

# Requirements
<ul>
        <li>Python <=3.10.8</li>
        <li>The Folder <kbd> Fine_Grain_Sentiment_Analyser</kbd> is the main folder must be placed where the file is importing it</li>
        <li>Do not Edit the folder <kbd> Fine_Grain_Sentiment_Analyser</kbd> until you are professional in</li>
</ul>

# How to setup
here . (dot) represents root folder of this project
<ol>
        <li>First Clone this repository</li>
        <li>And The Traverse to root Folder</li>
        <li>Now install all the modules written in requirement.txt or from root folder run <kbd>pip install -r requirements.txt</kbd></li>
        <li>An <kbd>Text.py</kbd> named file is given in root folder which have driver code for model</li>
</ol>
<hr>

# Driver code
```python
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
preprocessed_input=textInputProcessor.getPreprocessTheArray(text_input)

# generating the result
sentiment_result=textSentimentAnalyser.predict(preprocessed_input)

print(sentiment_result)
```
<p>output</p>
<pre>
> python Test.py
desire
</pre>

<hr>

# Required folder Structure
if you are a beginner the please follow the following path Structure

<pre>
📦Your_Project_Path
|------📦Fine_Grain_Sentiment_Analyser
|------┣--------📂Analyser
|------┃--------┣--------📜TextProcessor.py
|------┃--------┣--------📜Variables.py
|------┃--------┗--------📜__init__.py
|------┣--------📂Model
|------┃--------┣--------📜Model.joblib
|------┃--------┗--------📜Vectorizer.pickle
|------┗--------📜__init__.py
|------Code.py (This Code will have Driver code for this model)
</pre>

<hr>

# Data used for This Model
The following table is shows the all the description of the dataset used for this model building

<div itemscope itemtype="http://schema.org/Dataset">
  <table>
    <tr>
      <th>property</th>
      <th>value</th>
    </tr>
    <tr>
      <td>name</td>
      <td><code itemprop="name">GoEmotions</code></td>
    </tr>
    <tr>
      <td>description</td>
      <td><code itemprop="description">GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The emotion categories are _admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise_.</code></td>
    </tr>
    <tr>
      <td>sameAs</td>
      <td><code itemprop="sameAs">https://github.com/google-research/google-research/tree/master/goemotions</code></td>
    </tr>
    <tr>
      <td>citation</td>
      <td><code itemprop="citation">https://identifiers.org/arxiv:2005.00547</code></td>
    </tr>
    <tr>
      <td>provider</td>
      <td>
        <div itemscope="" itemtype="http://schema.org/Organization" itemprop="provider">
          <table>
            <tbody><tr>
              <th>property</th>
              <th>value</th>
            </tr>
            <tr>
              <td>name</td>
              <td><code itemprop="name">Google</code></td>
            </tr>
            <tr>
              <td>sameAs</td>
              <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/Google</code></td>
            </tr>
          </tbody></table>
        </div>
      </td>
    </tr>
  </table>
</div>

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2005.00547,
    doi = {
        10.48550/ARXIV.2005.00547
    },
    
    url = {
        https://arxiv.org/abs/2005.00547
    },
    
    author = {
        Demszky, Dorottya and Movshovitz-Attias, 
        Dana and Ko, Jeongwoo and Cowen, 
        Alan and Nemade, 
        Gaurav and Ravi, 
        Sujith
    },
    
    keywords = {
        Computation and Language (cs.CL), 
        FOS: Computer and information sciences, 
        FOS: Computer and information sciences
    },
    
    title = {
        GoEmotions: A Dataset of Fine-Grained Emotions
    },
    
    publisher = {
        arXiv
    },
    
    year = {
        2020
    },
    
    copyright = {
        arXiv.org perpetual, 
        non-exclusive license
    }
  }
  

