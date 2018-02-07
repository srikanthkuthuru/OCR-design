# OCR-design
1. Download Data from https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset
2. An end-to-end OCR system should be able to do the following:
	- Detect the location of words/characters on a scanned page [Localization]
	- Recognize those characters and assign them labels [Recognition]
	- Do error correction using language models, for example: find the closest word from a dictionary of possible words from literature. [NLP stuff]
	- Build image preprocessing models to improve robustness of the classifier. [I am not sure whether this is necessary or not!! CNNs can learn some standard preprocessing filters like edge enhancement filter etc.]



Thanks to authors in this paper who provided the data on UCI repository: S. Acharya, A.K. Pant and P.K. Gyawali Deep Learning Based Large Scale Handwritten Devanagari Character Recognition,In Proceedings of the 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA), pp. 121-126, 2015.

-Srikanth Kuthuru