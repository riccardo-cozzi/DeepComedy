import pyphen
import numpy as np

class ComedyTokenizer:
    
    
    def __init__(self, dictionary=None, synalepha=False, use_tercets=False):
        self.dictionary = dictionary
        self.synalepha = synalepha
        self.use_tercets = use_tercets
        self.vowels = vowels = ["a", "e", "i", "o", "u"]
        self.SYN = "~"
        self.SEP = "<S>"
        self.EOV = "</V>"
        self.EOT = "</T>"
        self.SOV = "<V>"
        self.SOT = "<T>"
        self.tokens = [self.SEP, self.EOV, self.EOT, self.SOV, self.SOT, self.SYN]
                       
    @staticmethod
    def from_dataframe(dataframe, synalepha=False, use_tercets=False):
        """
        Build a tokenizer based on an hyphenation dictionary stored in a pandas DataFrame.
        The dataframe must be indicised by words and contain a column called "hyphenation" 
        containing the hyphenation of the word
        """
        loaded_dictionary = dataframe.to_dict()["hyphenation"]
        return ComedyTokenizer(dictionary=loaded_dictionary, 
                                synalepha=synalepha, 
                                use_tercets=use_tercets)
        
        
    def isolate_punctuation(self, word):
        word = word.replace(".", " . ")
        word = word.replace(",", " , ")
        word = word.replace("?", " ? ")
        word = word.replace("!", " ! ")
        word = word.replace(":", " : ")
        word = word.replace(";", " ; ")
        word = word.replace('"', ' " ')    
        #word = word.replace("'", " ' ")    

        return word
    
    def remove_punctuation(self, word):
        word = word.replace(".", "")
        word = word.replace(",", "")
        word = word.replace("?", "")
        word = word.replace("!", "")
        word = word.replace(":", "")
        word = word.replace(";", "")
        word = word.replace('"', '')    
        #word = word.replace("'", " ' ")    

        return word



    def hyphenate(self, word):

        if len(word) == 1: return word

        try:
            if self.dictionary == None:
                raise Exception("Standard hyphenation selected.")
            
            # this may raise an exeption when the word is not in the dictionary. 
            # if so, the normal hyphenation is used. Alternatively, one can firstly chek
            # if the word is present and if yes return the hyphenation, otherwise, proceed.
            # In this way, instead, we prevent to look for the word in the dictionary a second time
            return self.dictionary[word] 

        except:
            import pyphen
            dic = pyphen.Pyphen(lang='it')
            word = dic.inserted(word)
            

            # if the first letter is a vowel, followed by a single consonant, then split the first syllable in two parts
            syl_1 = word.split("-")[0]
            if ((len(syl_1) >= 3)
                and (syl_1[0] in self.vowels)   
                and (syl_1[1] not in self.vowels) 
                and (syl_1[2] in self.vowels)):
                new_word = syl_1[0] + "-" + syl_1[1:]
                rest = word[len(syl_1):]
                #print("rest of the word: ", rest)
                for syl in rest:
                    new_word = new_word + syl
                return new_word
            return word
        
    
    
    def tokenize_phrase(self, phrase, count_syllables=False):
        phrase = phrase.replace("\n", "")
        phrase = [self.hyphenate(word) for word in phrase.split(" ")]
        phrase = self.SOV + " " + (" " + self.SEP + " ").join(phrase) + " " + self.EOV
        phrase = self.isolate_punctuation(phrase) # use this for make the punctuation marks be separated by SEP
        phrase = phrase.replace("-", " ")
        phrase = ' '.join([token for token in phrase.split(" ") if token != ""])
        if self.synalepha:
            phrase = self.apply_synalepha(phrase)
            
        
        if count_syllables:
            count = len([token for token in phrase.split(" ") if token not in self.tokens])
            return phrase, count
        else:
            return phrase
    
    
    
    def tokenize_text(self, text, use_tercets=None):
        if use_tercets == None:
            use_tercets = self.use_tercets
        tokenized = [self.SOT] if use_tercets else []
        
        for line in text:
            if line != "\n":
                line = self.tokenize_phrase(line)
                
                if use_tercets and tokenized[-1] == self.SOT:
                    tokenized[-1] += " " + line
                else:    
                    tokenized.append(line)
            else:
                if use_tercets:
                    tokenized.append(self.SOT)

        return np.array(tokenized)
    
    
    
    def apply_synalepha(self, string):
        new_tokens = []
        tokens = string.split(" ")
        i = 0
        while i < len(tokens):
            #print(tokens[i], "\t", new_tokens) ####### debug 
            # backward checking
            if (tokens[i][0] in self.vowels and i > 1):                             # the current token starts with a vowel
                if (new_tokens[-1] == self.SEP and new_tokens[-2][-1] in self.vowels):   # and the last added token in a separator and the token before ends with a vowel 
                    new_tokens[-2] += (self.SYN + tokens[i])                        #   add the token to the one before the separator
                    new_tokens = new_tokens[:-1]                                    #   remove the last separator
                
                elif (new_tokens[-1] != self.SEP):
                    if (tokens[-1] == self.SEP and new_tokens[-2][-1] in self.vowels): # the last added token is not a separator and it ends with a vowel
                        new_tokens[-1] += (self.SYN + tokens[i])                  #   add the new token to the last one
                    else:
                        new_tokens.append(tokens[i])
                else:
                    new_tokens.append(tokens[i])
                i += 1
                
            # forward checking
            elif (tokens[i][-1] in self.vowels                    # the current token ends with a vowel
                  and tokens[i+1] == self.SEP                          # and the next is a separator
                  and tokens[i+2][0] in self.vowels):             # and the successor starts by vowel
                new_tokens.append(tokens[i]+self.SYN+tokens[i+2]) #   add the next two wokens (without space) as synalepha
                i += 3
            
            else:
                new_tokens.append(tokens[i])
                i += 1
            
        return ' '.join(new_tokens)
    
    
    
    def remove_synalepha(self, string):
        return string.replace(self.SYN, self.SEP)
    
    
    def clear_text(self ,string):
        s = self.remove_synalepha(string)
        s = (s
             .replace(" ", "")
             .replace(self.SEP, " ")
             .replace(self.SOV, "")
             .replace(self.EOV, "\n")
             .replace(self.EOT, "\n\n"))
        
        return s