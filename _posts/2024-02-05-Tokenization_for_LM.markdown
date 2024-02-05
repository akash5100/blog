---
title: Tokenize & Numericalize
categories: deeplearning
---

Here are the main steps for language modeling:
1. Tokenization-- Converting Text into list of words (creating the vocab)
2. Numericalization-- Converting each word in vocab to number, by replacing them with their indices (simple!)

The next steps are: Language Model Data Creation (X&Y) and Language Model Creation

> **Jargon: Token** <br/> One element of a list created by the tokenization process. It could be a word, part of a word (a subword), or a single character.

To split the text into words, and to make the vocab, we need to convert text to words.

"This is an example." -> 'this', 'is', 'an', 'example', '.'

But there are different problem with this, text has lot of details. For instance, what if we have a chemistry long organic compound, oxybenzosomethingpentane, or some chinese font or japanese that don't use bases at all, or in english, a word like "don't"? German and Polish languages can be made so long by concatenating small pieces. How would you split it? 

There is no one correct answer to these question, there is no one approach to tokenization, so we use 3 main techniques:
1. Word based: what we saw above
2. subword based: "occasion" -> 'oc ca sion' (it splits words into more smaller parts)
3. character based: 'akash' -> a k a s h (individual character)
 
**SpaCy** is a opensource tokenization library. 

```py
spacy(['The U.S. dollar $1 is $1.00.'])
['The','U.S.','dollar','$','1','is','$','1.00','.']
```

Fastai has some rules, that it puts on top of this spacy tokens. They put some tokens starting with 'xx', these are special tokens. Pytorch names special tokens like '<something>'.

The most common item is, `xxbos` and `xxeos`, is beginning of stream and end of stream. 

`xxmaj` replaces the capitals letters, like

'L' -> 'xxmaj', 'l'

`xxrep` replaces repeating words,

'!!!' -> 'xxrep', '3', '!' (3 * !)

some other rules are,
- replacing html to text 
- replacing repeating words
- replacing useless spaces, removes repeating spaces
- wrapping spaces aroung '/' and '#'.
    '#' -> ' # ' & '/' -> ' / '
- lowercasing all character
- adding 'bos' and 'eos' at the beginning and end of a complete sequence.

here is a useful link: [text_proc_rules](https://github.com/fastai/fastai/blob/master/fastai/text/core.py#L23)

-----------------
<br/> *Subword Tokenization*

In addition to the word tokenization approach seen in the preceding section, another popular tokenization method is subword tokenization. Word tokenization relies on an assumption that spaces provide a useful separation of components of meaning in a sentence. However, this assumption is not always appropriate. For instance, consider this sentence: 私の名前はアカシュ・ヴェルマです ('My name is Akash Verma' in Japanese). That’s not going to work very well with a word tokenizer, because there are no spaces in it! Languages like Chinese and Japanese don’t use spaces, and in fact they don’t even have a well-defined concept of a “word.” Other languages, like Turkish and Hungarian, can add many subwords together without spaces, creating very long words that include a lot of separate pieces of information.

To handle these cases, its best to use subword tokenization. This proceeds in two steps:
1. Find most commonly occuring groups of letters, these became vocab.
2. tokenize the corpus using this vocab.

When using the library like Spacy or Fastai, we instantiate our subword tokenizer with what size of vocab it should be. --> `SubwordTokenizer(vocab_sz=2000)`. We want to create vocab for our text corpus, so passing our text corpus and 'train' it. --> `setup(txts)`. This creates vocab for our text corpus and then we can use that trained subwordtokenizer to tokenize things!

The length of each token depends on the size of vocab, if we created a vocab of size smaller, token size is smaller too that means it will break a single word into many words. Whereas if the vocab size to too big, it will consider a complete word as a token.

small vocab size -> 'y o u d is c over ed' -> more tokens

large vocab size -> 'you discover ed' -> less tokens

Picking a vocab size represents a compromise, a larger vocab size means fewer tokens per sentence, which means faster training and less memory and less state for the model to remember. But on the downside, it means larger embedding matrices, which require more data to learn.

"more data to learn", meaning, for example, if the text contains a name 'akash' this would be considered as one token, whereas if the vocab size is small, it would be 'ak ash' and 'ash' could be an already repeating word in the corpus. To solve this problem, we can replace words like this (which occurs rarely) with an unknown special token, `<UKN>` or `xxukn`. This can reduce the embedding matrix size. But then, this means there is also less data for the new rare words (that we replaced with ukn flag), now how can we learn about those rare word in the corpus?

This last issue is better handeled by setting a minimum frequence threshold; for example, min_freq=3 means that any word appearing fewer than three times is replaced with `xxukn`.

I learned creating RNN from scratch using this dataset: [Human Numbers dataset](https://www.leebutterman.com/2020/09/30/human-numbers-100k.html)

It contains 2 file, train.txt and valid.txt. Merging the two to make it a big corpus. and use it for language modeling.

```py
with open(path/'train.txt') as f:
    dat = f.read()
    print(dat[:52])
    
print('\n')
with open(path/'valid.txt') as f:
    dat2 = f.read()
    print(dat2[:52])
    
dat += dat2

# outputs:
# one 
# two 
# three 
# four 
# five 
# six 
# seven 
# eight 
# nine


# eight thousand one 
# eight thousand two 
# eight thousa
```

We will take all those lines and concatenate them in one big stream. From one number to next, we use "." as a seperator.

This is how our stream will look like: "one . two . three . [...]"

```py
text = ". ".join(dat.split("\n"))
tokens = text.split(" ")
tokens.remove("") # how does this get in?
# tokens will look like:
# ["one", ".", "two", ".", [...]]
```


We need to create vocab, creating a frequency map of all the tokens we have:

```py
freq = list({i for i in tokens})
freq = {w: 0 for w in freq}
for i in tokens: freq[i]+=1
```

Now, we have a vocab. Converting them into number. Because fastai numericalize the tokens by ordering them with there frequency of occurance (if a token occurs 100000 times it will be in 0th index of our vocab) I tried to replicate that:

```py
freq1 = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
vocab = list(freq1); vocab[:3]
word2idx = {w: i for i,w in enumerate(freq1)}; word2idx
# a map to convert tokens to numbers
{'.': 0,
 'hundred': 1,
 'thousand': 2,
 'six': 3,
 'five': 4,
 'two': 5,
 'nine': 6,
 'four': 7,
 'seven': 8,
 'three': 9,
 'one': 10,
 'eight': 11,
 'eighty': 12,
 'forty': 13,
 'ninety': 14,
 'twenty': 15,
 'sixty': 16,
 'thirty': 17,
 'fifty': 18,
 'seventy': 19,
 'sixteen': 20,
 'fourteen': 21,
 'eleven': 22,
 'thirteen': 23,
 'eighteen': 24,
 'seventeen': 25,
 'ten': 26,
 'fifteen': 27,
 'nineteen': 28,
 'twelve': 29}
```

*Numericalize*-- Replacing the tokens with their indices.

```py
nums = [word2idx[i] for i in tokens] 
nums[:20] # our first 20 tokens.
[10, 0, 5, 0, 9, 0, 7, 0, 4, 0, 3, 0, 8, 0, 11, 0, 6, 0, 26, 0]
```

We have our dataset ready!