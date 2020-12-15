#  -*- coding: utf-8 -*-

# import required packages
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
import nltk
from bs4 import BeautifulSoup
from treelib import Tree
import re
import hashlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
from utils import *


class FacultyPagesFilteredSpider(scrapy.Spider):
    name = 'faculty_pages_filtered'
    allowed_domains = ['cmu.edu',
                       'cornell.edu', 'washington.edu',
                       'gatech.edu', 'princeton.edu', 'utexas.edu',
                       'illinois.edu','berkeley.edu'
                       'mit.edu', 'stanford.edu']
    count = 0
    record = {}
    start_urls =  ['https://www.cmu.edu/',
                  'https://www.cornell.edu/',
                  'https://www.washington.edu/', 'https://www.gatech.edu/',
                  'https://www.princeton.edu/', 'https://www.utexas.edu/',
                  'https://illinois.edu/', 'https://www.berkeley.edu/',
                  'https://www.mit.edu/', 'https://www.stanford.edu/']

    exclude_words = ['news', 'events', 'publications', 'pub', 'gallery', 
                     'category', 'courses', 'students', 'references', 
                     'reference', 'software', 'softwares', 'tags', 
                     'tutorials', 'workshop', 'festival', 'admissions', 
                     'exhibitions', 'alumni', 'lectures', 'undergraduate', 
                     'about', 'history', 'awards', 'ranking', 'enrollment', 
                     'graduate', 'archive', 'stories', 'post', 'pages', 
                     'magazine', 'curriculum', '404', 'faqs', 'engage', 
                     'campaign', 'career', 'resources', 'services', 
                     'network', 'security', 'donate', 'giving', 'finance', 
                     'forms', 'policies', 'policy', 'alphabetical', 'summer', 
                     'winter', 'spring', 'autumn', 'fall', 'health', 'facilities', 
                     'facility', 'wp', 'information', 'general', 'catalog', 
                     'guides', 'library', 'publish', 'blog', 'collection', 
                     'share', 'search', 'periodicals', 'bookstore', 'store', 
                     'product', 'organisation', 'webstore', 'funding', 'pdf']


    rules = [Rule(LinkExtractor(unique=True), callback='parse', follow=True)]
    #count_limits = {"page_count": 200, "item_count": 200}

    def __init__(self):
        
        self.tree = Tree()
        self.tree.create_node("root", "root")
        self.tree.create_node("unknown", "unknown", parent="root")
        
        self.bio_identifier = BioIdentifier(model="bio-model")

        for dom in self.allowed_domains:
            domain = dom.split('.')[0]
            if not os.path.exists('Crawled_Data'):
                os.makedirs('Crawled_Data')

            folder_name = 'Crawled_Data/'+domain.capitalize() + '_University_Files'
            self.record[domain] = 0
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

    def parse(self, response):
        
        matched_domain = [x for x in self.allowed_domains if x in response.url]
        if len(matched_domain) > 0:
            domain = matched_domain[0].split('.')[0]
            
            folder_name = 'Crawled_Data/'+domain.capitalize() + '_University_Files'

            self.record[domain] = self.record.get(domain, 0) + 1
            
            if self.record[domain]%50 == 0:
                print('\n Crawled {} Bio-pages of {} University ...'.format(self.record[domain], domain.capitalize()))
                self.tree.save2file(folder_name+"/00__"+str(self.record[domain])+"_tree.txt")

            isBio = self.bio_identifier.is_bio_html_content(response.xpath('//*').get())
            
            if isBio:
                text = BeautifulSoup(response.xpath('//*').get(), features="html.parser").get_text()
                tokens = nltk.word_tokenize(text)
                normalized_text = ' '.join([word for word in tokens if word.isalnum()])
                normalized_text += '\n'+response.url
                
                hash_text = hashlib.md5(response.url.encode()) 
                file_name = hash_text.hexdigest()

                with open(folder_name+"/"+file_name+".txt", "w", encoding="utf-8") as file:
                    file.write(normalized_text)
                    
            
            AllLinks = LinkExtractor(allow_domains = domain+'.edu', unique=True).extract_links(response)

            for n, link in enumerate(AllLinks):
                if not any([x in link.url for x in self.exclude_words]):
                    if self.tree.get_node(link.url) == None:
                        referer = response.request.headers.get('Referer', None)

                        if referer == None:
                            self.tree.create_node(link.url, link.url, parent='root')
                        else:
                            referer = referer.decode("utf-8")
                            if self.tree.contains(referer):

                                self.tree.create_node(link.url, link.url, parent=referer)
                            else:
                                self.tree.create_node(link.url, link.url, parent='unknown')

                        yield scrapy.Request(url=link.url, callback = self.parse)

