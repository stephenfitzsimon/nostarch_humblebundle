import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

LINES = ['-', ':', '--'] # to be used for making the line graphs

def main():
    # get the texts and store in a dictionary
    strings_by_author = dict()
    # load each string into the dictionary as a string, use key author name
    strings_by_author['doyle'] = text_to_string('hound.txt')
    strings_by_author['wells'] = text_to_string('war.txt')
    strings_by_author['unknown'] = text_to_string('lost.txt')

    # ensure things go as plan
    print(strings_by_author['doyle'][:300])

    # will split the .txt into words and return as a list, with key 
    # as the author name
    words_by_author = make_word_dict(strings_by_author)
    # returns the length of the shorted corpus
    len_shortest_corpus = find_shortest_corpus(words_by_author)
    word_length_test(words_by_author, len_shortest_corpus)
    stop_words_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    vocab_test(words_by_author)
    jaccard_test(words_by_author, len_shortest_corpus)

# loading the text and building a word dictionary
def text_to_string(filename):
    """ Read a text file and return a string """
    #use with to keep the file open only for this context
    with open(filename) as infile:
        # return read file and close contest
        return infile.read() 

def make_word_dict(strings_by_author):
    """ Return a dictionary of tokenized words by corpus author """
    # to store the words
    words_by_author = dict()
    # loop through all the authors
    for author in strings_by_author:
        # get the tokens from the nltk tekenizer
        # this is just a list of the words used by the author, note that is also 
        # tokenizes punctuation
        tokens = nltk.word_tokenize(strings_by_author[author])
        # make them all lower case.  This also filters out punctuation and hyphented words
        # by using str.isalpha()
        words_by_author[author] = ([token.lower() for token in tokens if token.isalpha()])
    # return the dictionary
    return words_by_author

def find_shortest_corpus(words_by_author):
    """" Return the length of the shortest corpus """
    #holds the length of the dictionaries passed by author
    word_count = []
    for author in words_by_author:
        #get the length of the words for each author
        word_count.append(len(words_by_author[author]))
        #print to user
        print(f"\nNumber of words for {author} = {len(words_by_author[author])}\n")
    #get shortest corpus
    len_shortest_corpus = min(word_count)
    #print to user
    print("Length of shortest corpus = {len_shortest_corpus}\n")
    #return to call
    return len_shortest_corpus

def word_length_test(words_by_author, len_shortest_corpus):
    """ Plot word length freq by author, truncated to shortest corpus length """
    by_author_length_freq_dict = dict()
    #set figure to 1 as there will be multiple figures
    plt.figure(1)
    #turns on interactive plot mode
    plt.ion()
    for i, author in enumerate(words_by_author):
        # get the length of each word in the lexicon, only go up to the length of the shortest corpus
        word_lengths = [len(word) for word in words_by_author[author][:len_shortest_corpus]]
        # extract the data so that is can be plotted
        by_author_length_freq_dict[author] = nltk.FreqDist(word_lengths)
        # limit to words that are no more than 15 chars long, using a seperate linestyle for 
        # each author, and set the label to the author, and title the plot
        by_author_length_freq_dict[author].plot(15, linestyle = LINES[i], label=author, title='Word Length')
    #display legend
    plt.legend()
    #show the plot
    #plt.show()

def stop_words_test(words_by_author, len_shortest_corpus):
    """ Plot stopwords freq by author, truncated to shortest corpus length """
    stopwords_by_author_freq_dist = dict()
    #will be second figure plotted
    plt.figure(2)
    #get a list of stop words as a set (increases speed)
    stop_words = set(stopwords.words('english'))
    print('Number of stopwords = {}\n'.format(len(stop_words)))
    print('Stopwords = {}\n'.format(stop_words))
    for i, author in enumerate(words_by_author):
        #only get the word if it's a stop word
        stopwords_by_author = [word for word in words_by_author[author][:len_shortest_corpus] if word in stop_words]
        stopwords_by_author_freq_dist[author] = nltk.FreqDist(stopwords_by_author)
        # plot the frequency
        stopwords_by_author_freq_dist[author].plot(50, label=author, linestyle=LINES[i], title = '50 most common stopwords')
    plt.legend()
    #plt.show()

def parts_of_speech_test(words_by_author, len_shortest_corpus):
    """" Plot author use of parts-of-speech """
    by_author_pos_freq = dict()
    plt.figure(3)
    for i, author in enumerate(words_by_author):
        pos_by_author = [pos[1] for pos in  nltk.pos_tag(words_by_author[author][:len_shortest_corpus])]
        by_author_pos_freq[author] = nltk.FreqDist(pos_by_author)
        by_author_pos_freq[author].plot(35, label=author, linestyle=LINES[i], title = 'Parts of Speech')
    plt.legend()
    plt.show()

def vocab_test(words_by_author):
    """" Compare author vocabularies using chi^2 statistical test """
    chisquared_by_author = dict() #to hold the calculated values
    for author in words_by_author:
        if author != 'unknown':
            #total corpus
            combined_corpus = (words_by_author[author] + words_by_author['unknown'])
            #specific author corpus
            author_proportion = (len(words_by_author[author])/len(combined_corpus))
            #get the fequency distribution
            combined_freq_dist = nltk.FreqDist(combined_corpus)
            #test only the 1000 most common words
            most_common_words = list(combined_freq_dist.most_common(1000))
            chisquared = 0
            # calculate the chisquared
            for word, combined_count in most_common_words:
                observed_count_author = words_by_author[author].count(word)
                expected_count_author = combined_count * author_proportion
                chisquared += ((observed_count_author - expected_count_author)**2 / expected_count_author)
                chisquared_by_author[author] = chisquared
            print('Chi-squared for {} = {:.1f}'.format(author, chisquared))
    # the lower the chisquared the more similar 
    most_likely_author = min(chisquared_by_author, key=chisquared_by_author.get)
    print('Most-likely author by vocabulary is {}\n'.format(most_likely_author))

