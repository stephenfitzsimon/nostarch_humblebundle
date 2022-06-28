# nostarch_humblebundle

Repository for tutorials from the books included in the NoStarch Humble Bundle

## Real World Python
- `bayes.py` : searching a map using OpenCV to explore Baye's theorem.  Uses a class to organize code and help with program flow.  In addition, OpenCV is used to interact with an image file.
- `bayes_smarter_searches.py` : Does not repeat the search in an area that already was searched.
    - Right now, it uses filter() to filter out all the coordinates already searched.  There might be a better way of doing this if the coordinates are stored in a numpy array.  Then `settdiff1d` could be used to filter, which might be more efficient.
- `stylometry.py` : use frequency analysis to determine the author of a text.  Uses `nltk` NLP module to tokenize texts and then compare them using differeny measures; including, word length, stop word frequency, and part-of-speech frequency.
