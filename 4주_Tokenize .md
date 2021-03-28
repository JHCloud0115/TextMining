```python
# 필요한 nltk library download
import nltk
nltk.download('punkt')
nltk.download('webtext')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\juhee\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package webtext to
    [nltk_data]     C:\Users\juhee\AppData\Roaming\nltk_data...
    [nltk_data]   Package webtext is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\juhee\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\juhee\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\juhee\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    




    True



## 문장 토큰화(sentence tokenize)

토크나이즈 실시 (토큰 : 관심 있는 단위)  
즉, 문장 토큰화란 주어진 paragraph 혹은 document로 부터 setence 단위로 자르는 것 



```python
para = "Hello everyone. It's good to see you. Let's start our text mining class!"
```


```python
#영어가 사용된 문장 실시 

from nltk.tokenize import sent_tokenize
print(sent_tokenize(para)) #주어진 text를 sentence 단위로 tokenize함. 주로 . ! ? 등을 이용
```

    ['Hello everyone.', "It's good to see you.", "Let's start our text mining class!"]
    


```python
#프랑스어가 사용된 문장 실시 

paragraph_french = """Je t'ai demandé si tu m'aimais bien, Tu m'a répondu non. 
Je t'ai demandé si j'étais jolie, Tu m'a répondu non. 
Je t'ai demandé si j'étai dans ton coeur, Tu m'a répondu non."""

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
print(tokenizer.tokenize(paragraph_french))
```

    ["Je t'ai demandé si tu m'aimais bien, Tu m'a répondu non.", "Je t'ai demandé si j'étais jolie, Tu m'a répondu non.", "Je t'ai demandé si j'étai dans ton coeur, Tu m'a répondu non."]
    


```python
para_kor = "안녕하세요, 여러분. 만나서 반갑습니다. 이제 텍스트마이닝 클래스를 시작해봅시다!"
```


```python
print(sent_tokenize(para_kor)) 
```

    ['안녕하세요, 여러분.', '만나서 반갑습니다.', '이제 텍스트마이닝 클래스를 시작해봅시다!']
    

한글에 대한 사전학습된 모델은 없지만 setence tokenizer 실시해본 결과 잘 작동되는 것 확인 

## 단어 토큰화 (word tokenize)


```python
#word_tokenize 사용

from nltk.tokenize import word_tokenize
#주어진 text를 word 단위로 tokenize함
print(word_tokenize(para))
```

    ['Hello', 'everyone', '.', 'It', "'s", 'good', 'to', 'see', 'you', '.', 'Let', "'s", 'start', 'our', 'text', 'mining', 'class', '!']
    


```python
#WordPunctTokenizer사용 

from nltk.tokenize import WordPunctTokenizer  
print(WordPunctTokenizer().tokenize(para))
```

    ['Hello', 'everyone', '.', 'It', "'", 's', 'good', 'to', 'see', 'you', '.', 'Let', "'", 's', 'start', 'our', 'text', 'mining', 'class', '!']
    

<둘 다 사용해본 결과>  
 서로 다른 알고리즘 기반으로 생성되었기 때문에 WordPunctTokenizer는 It's를's를 It, ', s의 세 토큰으로 분리하는 것을 확인할 수 있다. (어느 것을 사용해도 상관은 없음)


```python
print(word_tokenize(para_kor))
```

    ['안녕하세요', ',', '여러분', '.', '만나서', '반갑습니다', '.', '이제', '텍스트마이닝', '클래스를', '시작해봅시다', '!']
    

한글은 조사, 어간, 어미가 있기 때문에 잘 되지 않는 것을 확인할 수 있음. 형태소 단위로 나누어야하지만, word_tokenizer는 어절단위로 구분해줌 ---> 형태소 분석이 필요한 이유 

## 정규표현식을 이용한 토큰화

**정규표현식**  
정규표현식은 패턴 표현 위해 메타 문자 사용  

1. 메타 문자는 ``문자 클래스 []``  
문자 클래스는 그 사이에 들어있는 문자와 매칭을 시켜서 클래스 안에 하나라도 일치하는 문자가 있다면 가져온다 


```python
import re
re.findall("[abc]", "How are you, boy?")
```




    ['a', 'b']




```python
#숫자를 찾고 싶을 때

re.findall("[0123456789]", "3a7b5c9d") #숫자 다 쓰기 싫을땐 [0-9] '-'사용하면 됌
                                       #알파벳도 마찬가지로 [a-zA-Z] 
```




    ['3', '7', '5', '9']




```python
#알파벳+숫자+_[a-zA-Z0-9_]
re.findall("[a-zA-Z0-9_]", "3a 7b_ '.^&5c9d")
```




    ['3', 'a', '7', 'b', '_', '5', 'c', '9', 'd']




```python
#[\w]로 대체 가능 --> 위의 결과와 같은 결과 출력 확인0
re.findall("[\w]", "3a 7b_ '.^&5c9d")
```




    ['3', 'a', '7', 'b', '_', '5', 'c', '9', 'd']



2. ``+ 메타문자`` 사용    
한 번 이상의 반복 의미 


```python
re.findall("[_]+", "a_b, c__d, e___f")
```




    ['_', '__', '___']




```python
re.findall("[\w]+", "How are you, boy?")
# -->split()은 특수문자 제거해주지 않지만 [\w]는 가능
```




    ['How', 'are', 'you', 'boy']



2-1. ``{}``사용하면, 정확한 반복횟수 지정 가능


```python
re.findall("[o]{2,4}", "oh, hoow are yoooou, boooooooy?")
```




    ['oo', 'oooo', 'oooo', 'ooo']



위 예는 2,4번 반복횟수 지정한 사례로 oh의 o는 한번이기 때문에 검색되지 않음. 반면 boooooooy는 o가 7개이기 때문에 최대개수인 앞 네 개가 매칭되고 남은 뒤의 세 개가 또 매칭되었다. 즉 'oo' <-- hoow / 첫째 'oooo' <-- yoooou / 둘째 'oooo'<-- boooo / 'ooo'<-- oooy에서 매칭


```python
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+") #regular expression(정규식)을 이용한 tokenizer
#단어단위로 tokenize \w:문자나 숫자를 의미 즉 문자나 숫자 혹은 '가 반복되는 것을 찾아냄
print(tokenizer.tokenize("Sorry, I can't go there."))
# can't를 하나의 단어로 인식
```

    ['Sorry', 'I', "can't", 'go', 'there']
    


```python
#3자 이상 가져오기 

text1 = "Sorry, I can't go there."
tokenizer = RegexpTokenizer("[\w']{3,}") 
print(tokenizer.tokenize(text1.lower()))
```

    ['sorry', "can't", 'there']
    
