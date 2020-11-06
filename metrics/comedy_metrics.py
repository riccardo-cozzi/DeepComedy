from nltk.metrics import edit_distance
import numpy as np
import re



TABLE_WIDTH = 100 # constant for printing "fill" characters

# returns the texts with no 
def extract_only_verses(canto):
  return [line for line in canto if line != "\n" and line != ""]


def remove_accents(string):
  string = re.sub(r"[àá]", "a", string)
  string = re.sub(r"[èé]", "e", string)
  string = re.sub(r"[ìí]", "i", string)
  string = re.sub(r"[òó]", "o", string)
  string = re.sub(r"[ùú]", "u", string)
  return string


def extract_vocabulary(text):
  if type(text) == list:
    s = set()
    for string in text:
      s = s.union(extract_vocabulary(string))
    vocab = s
    return vocab
  else: 
    vocab = text.lower()
    vocab = remove_accents(vocab)
    vocab = text.replace("\n", " ") 
    vocab = re.sub(r"[^a-z\s]", " ", vocab)
    all_words = vocab.split(" ")
    unique_words = np.unique(all_words)
    vocab = set([r for r in unique_words if len(r) > 2 ])
    return vocab


def count_verses(canto):
  return len(extract_only_verses(canto))


def evaluate_structure(canto, final_single_verse=True, verbose=False):
  """
  canto :\n\tlist of strings, each one representing a verse \n
  final_single_verse (default=True):\n\tif False, return the maximum score (1.0) iff all the tercets have 3 verses. \n
  Otherwise, the maximum score is reached iff the canto ends with a single verse \n
  verbose (default=False): \n\tif True, prints out the count of verses, groups and tercets for each verse \n
  Return\n\tstructuredness score in [0, 1]
  """
  groups = 1
  tercets = 0
  verse_counter = 0
  if verbose: 
    print("{:>40}\t{:<10}  {:<10}  {:<15}".format("", "n. verse", "n. groups", "n. tercets"))
    print("-"*TABLE_WIDTH)

  # count groups of verces
  for i, verse in enumerate(canto):
    verse = verse.replace("\n", "")
    if not bool(re.search("[a-zA-z]", verse)):
      groups += 1
      if verse_counter == 3:
        tercets += 1
      verse_counter = 0
    else:
      verse_counter += 1 

    # count the final group which is ignored
    if i == len(canto)-1 and verse_counter == 3:
      tercets += 1

    if verbose: print("-{:<40}\t{:<10}  {:<10}  {:<15}".format(verse, verse_counter, groups, tercets))

  # check whether the final verse is in a single line
  if final_single_verse:
    correct = (not bool(re.search("[a-zA-z]", canto[-2])) and bool(re.search("[a-zA-z]", canto[-1])))
    if correct:
      tercets += 1
    else: 
      groups += 1
      # print("ultimo: [{}]".format(canto[-1]))
      # print("penultimo: [{}]".format(canto[-2]))
    if verbose: 
      print(" "*40 + " ")
      print("\t  single final verse:   {}\t\t{:<10}  {:<10}  {:<15}".format(correct, 
                                                                            verse_counter, 
                                                                            groups, 
                                                                            tercets))  
  if verbose: print("-"*TABLE_WIDTH) 

  return tercets / groups



# count number of hendecasyllables verses
def evaluate_hendecasyllables(canto, tokenizer, return_count=False, tolerance=1, verbose=False):
    verses = [line for line in canto if line != "\n" and line != ""]
        
    n_syllables = []
    for verse in [v for v in verses if v != ""]:
        verse = verse.replace("\n", "")
        if verse != "":
            verse = tokenizer.remove_punctuation(verse)
            sill, n = tokenizer.tokenize_phrase(verse, count_syllables=True)
            if verbose: print("{:40}{:80} ({})".format(verse, sill, n))
            n_syllables.append(n)
    
    n_hendecasillables = len([n for n in n_syllables if n in range(11-tolerance, 11+tolerance+1)])
    
    if verbose: 
        print("="*80)

    if return_count:
        return n_hendecasillables/len(verses), n_syllables
    else:
        return n_hendecasillables/len(verses)



def average_hendecasyllables(canto, tokenizer, tolerance):
  _, n_syllables = evaluate_hendecasyllables(canto=canto, tokenizer=tokenizer, tolerance=tolerance, return_count=True)
  return np.average(n_syllables)



def avg_rhyming_score(canto, tokenizer, return_n_rhymes=False, raw=True, verbose=False):
  if raw:
    finals = []
    verses = extract_only_verses(canto)
    for verse in verses:
      finals.append(tokenizer.tokenize_phrase(tokenizer.remove_punctuation(verse)).split(" ")[-2])
    
    # find chained rhyme pattern
    scores = []
    for i in range(0, len(verses)-4, 3):
      score = 0
      if finals[i] == finals[i+2]:
        score += 0.5
      if finals[i+1] == finals[i+3]:
        score += 0.5
      scores.append(score)
      if verbose and score != 1: print("\n{:50} ({})\n{:50} ({})\n{:50} ({})\n{:50} ({})\n- score: {}".format(
          verses[i], finals[i], verses[i+1], finals[i+1], 
          verses[i+2], finals[i+2], verses[i+3], finals[i+3],
          score
      ))


    if return_n_rhymes:
      return np.average(scores), len([s for s in scores if s == 1])
    else:
      return np.average(scores)

  else:
    return None


def ngrams_plagiarism(generated_text, original_text, n=4):
  """
  (credits: Luga Giuliani)
  """
  return -1
  # the tokenizer is used to remove non-alphanumeric symbols
  tokenizer = tfds.features.text.Tokenizer()
  original_text = tokenizer.join(tokenizer.tokenize(original_text.lower()))
  generated_text_tokens = tokenizer.tokenize(generated_text.lower())

  total_ngrams = len(generated_text_tokens) - n + 1
  plagiarism_counter = 0

  for i in range(total_ngrams):
      ngram = tokenizer.join(generated_text_tokens[i:i+n])
      plagiarism_counter += 1 if ngram in original_text else 0
  return 1 - (plagiarism_counter / total_ngrams)


def find_similar_words(word:str, vocabulary:set, verbose=False, return_best_distance=False):
  """
  Given a word, find the most similar words in the vocabulary. \n
  The similarity between words is computed by the 'edit (Levenshtein) distance'. \n
  If more than one word from the vocabulary has a same distance from the requested word, \n
  then a list containing them is returned. \n
  If #return_best_distance is True, then returns a tuple containing the best words and the best distance. 
  """
  if verbose: print(f"looking for words similar to '{word}'")
  # try normal search
  if word in vocabulary:
    if verbose: print(f"\t{(real_word, 0)} <-- match")
    return ([word], 0) if return_best_distance else [word]

  else:
    most_similar = []
    best_distance = len(word) # the distance between a word and an empty word is equal to the lenght of the word itself

    for real_word in vocabulary:
      dist = word_distance(word, real_word)
      if dist <= best_distance:
        if verbose: print(f"\t {real_word} ({dist})")
        if dist < best_distance:
          best_distance = dist
          most_similar = []
        most_similar.append(real_word)
    return (most_similar, best_distance) if return_best_distance else most_similar


def word_distance(a, b):
  from nltk.metrics import edit_distance
  d = edit_distance(a, b)
  return d



def incorrectness(words:set, real_words:set, verbose=False, return_match_ratio=False, plot_frequencies=False):
  """
  Measures the amount of incorrect words, with respect to the given set of real words. 
  If all the passed words exists in the real words set, then the returned value is 0, otherwise return a positive real number.
  The score is computed as a weighted average of the frequencies of the distances of the words from the real words. 
  A set of words each of which has the nearest word (in Levenshtein distance) at 1 or 2 is way better of another one
  whose nearest word is 10 points far from the most similar real word. 
  """
  if verbose: print("{}\n{:3}\t{:10}\t{:5}\t{}\n{}\n".format("="*40, "%", "WORD", "DIST", "SIMILAR TO", "="*40))
  n_real_words = len(words)
  # compute frequencies
  distances = []
  for i, my_word in enumerate(words):
    most_similar, distance = find_similar_words(word=my_word, 
                                                 vocabulary=real_words,
                                                 return_best_distance=True)
    distances.append(distance)
    
  # compute frequencies
  frequencies = dict(zip(np.unique(distances), [distances.count(d) for d in np.unique(distances)]))
  
  # add the zero-frequency if not present (in order to compute the correct words percentage)
  if 0 not in frequencies.keys(): frequencies[0] = 0
  
  if verbose: print("\n{}\n frequencies: {}".format("-"*40, dict(frequencies)))
  
  # computing correctness
  incorrectness = round(np.average(np.unique(distances),
                                 weights=[distances.count(d) for
                                          d in np.unique(distances)]), 2)
  
  # percentage of incorrect words
  ratio = 1 - round(frequencies[0] / len(words), 2)

  # print final results
  if verbose: 
    print(" match ratio:  {} %\t({} / {}) \n{}".format(ratio, frequencies[0],
                                                       len(words), "="*40))
    if distance != 0: 
      print("{:>3}\t{:15}\t{:<5}\t{}".format(round(i/n_real_words*100, 1),
                                             my_word, distance, most_similar))  
  # plot
  if plot_frequencies: 
    from matplotlib import pyplot as pyplot
    plt.bar(list(frequencies), [frequencies[key] for key in list(frequencies)])
    plt.show()

  return (incorrectness, ratio) if return_match_ratio else incorrectness
