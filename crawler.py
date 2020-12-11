import pathlib
import nltk
from tinydb import TinyDB
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
import uuid


class CrawlerSpider(CrawlSpider):
    name = 'Crawler'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.start_urls = [kw.get('start_url')]
        self.allowed_domains = [kw.get('allowed_domain')]
        self.base_url = kw.get('start_url')
        self._db = TinyDB(kw.get('db'))
        if kw.get('exclude'):
            self.rules = [(
                Rule(LinkExtractor(deny=kw.get('exclude'), unique=True), callback='parse', follow=True)
            )]
        else:
            self.rules = [(
                Rule(LinkExtractor(unique=True), callback='parse', follow=True)
            )]
        self._compile_rules()
        self._pages = pathlib.PurePath(pathlib.Path().absolute(), kw.get('output'))
        Path(self._pages).mkdir(parents=True, exist_ok=True)

    def parse(self, response):
        text = BeautifulSoup(response.xpath('//*').get(), 'html.parser').get_text().lower()
        tokens = nltk.word_tokenize(text)
        normalized_text = ' '.join([word for word in tokens if word.isalnum()])
        file_path = pathlib.PurePath(self._pages, "{}.html".format(uuid.uuid1()))
        self._db.insert({"url": response.request.url, "file": file_path.as_uri()})
        with open(file_path, 'w') as f:
            f.write(normalized_text)