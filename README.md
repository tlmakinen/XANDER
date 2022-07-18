# xander: X-ray ANomaly DEtectoR <img src="https://raw.githubusercontent.com/tlmakinen/xander/master/assets/images/xander-logo.png" alt="drawing" width="65"/> 
This code is part of an exploratory study for finding anomalies in the Chandra source catalog directly from time series data. Our goal is to automate otherwise serendipitous discovery with the help of deep learning architectures trained on existing preprocessed data. 

### experiment overview
<img src="https://raw.githubusercontent.com/tlmakinen/xander/master/assets/images/xander-flowchart.png" alt="drawing" width="700"/>

### toy model in Colab:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eQF63XlHaZs0haBTgscGyy804yNn1sTt?usp=sharing)
 This is a self-contained tutorial that can be run in-browser.

### graphnets implementation
The graph networks used to regress to summary statistics are implemented using the `jraph` and `dm-haiku` libraries. See the Colab demonstration for an implemented toy model. 
<img src="https://raw.githubusercontent.com/tlmakinen/xander/master/assets/images/gnnregress.png" alt="drawing" width="700"/>

### graph network algorithm animation

<img src="[https://raw.githubusercontent.com/tlmakinen/xander/master/assets/images/gnnregress.png](https://media1.giphy.com/media/qyVMnbeGkwJO8enqqw/giphy.gif?cid=790b76117bda8ebc917d894043d220659329f156c8b830d1&rid=giphy.gif)" alt="drawing" width="700"/>
