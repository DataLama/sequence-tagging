import re
import emoji
import unicodedata
from bs4 import BeautifulSoup
from soynlp.normalizer import repeat_normalize
    
class ZhPreprocessing:
    """All about chinese text preprocessing function in NLP"""
    def __init__(self):
        self.emoji = emoji.get_emoji_regexp()
        self.pattern = re.compile(f'[^ .,?!/@$%~％·∼()。、，《 》“”：\x00-\x7F\u4e00-\u9fff{self.emoji}]+') # 기호, 영어, 중국어, 이모티콘
        self.url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
        self.hashtag = re.compile(f'#([{self.emoji.pattern}\w-]+)')
        self.mention = re.compile(f'@([\w-]+)')
        self.image = re.compile(r'(\[image#0\d\])')
    
    def normalize_chinese_pattern(self, text:str) -> str:
        """영어, 중국어, 이모지, 특수기호를 제외한 모든 것을 제거함."""
        return self.pattern.sub('', text)
    
    def rm_url(self, text:str) -> str:
        return self.url.sub('', text)
    
    def rm_email(self, text:str) -> str:
        return self.email.sub('', text)
    
    def rm_emoji(self, text:str) -> str:
        return self.emoji.sub('', text)
    
    def rm_hashtag(self, text:str) -> str:
        return self.hashtag.sub('', text)
    
    def rm_mention(self, text:str) -> str:
        return self.mention.sub('', text)

    def rm_image(self, text:str) -> str:
        return self.image.sub('', text)
    
def normalize(text: str) -> str:
    # unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # eradicate html script
    html = re.compile("<(\"[^\"]*\"|'[^']*'|[^'\">])*>")
    if html.search(text) != None: # html js 처리
        soup = BeautifulSoup(text, "lxml")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()

    # chinese preprocessing format
    text = preprocessor.normalize_chinese_pattern(text)

    # normalize repeated pattern
    text = repeat_normalize(text, num_repeats=3)

    return text.strip()

preprocessor = ZhPreprocessing()

def preprocess(text):
    text = preprocessor.rm_url(text)
    text = preprocessor.rm_email(text)
    text = preprocessor.rm_mention(text)
    text = preprocessor.rm_image(text)
    text = preprocessor.rm_emoji(text)
    text = preprocessor.rm_hashtag(text)
    return text