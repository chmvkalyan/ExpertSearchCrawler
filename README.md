# ExpertSearch - Automated Crawler

The purpose of automated crawler is to improve upon the data collection for the ExpertSearch project (link). 

## Improvement
In the existing ExpertSearch project the database is collected manually as follows:

- Go to university's website
- Go to department's webpage
- Go to faculty directory page
- Extract links of faculty bio pages
- Process and download from faculty bio page

This process is manual, strenous, time-consuming and not dynamic.

The proposed method is to automate these tasks in the ExpertSearch System.

## Setup
~~~~
# Install scrapy
pip install scrapy

# Creating Conda environment
conda env create -f env.yml
# Activating the environment
conda activate BioPageClassifier

# Downloading NLTK Corpora
python -m nltk.downloader all
# Downloading spaCY model
python -m spacy download en_core_web_sm

wget https://drive.google.com/file/d/1NShUBtE248LN_L1zzyGbK__4I60bkk0R/view?usp=sharing
[ ! -d "bio-model" ] && unzip bio-model.zip 

# Execute the webcrawler
scrapy crawl faculty_pages_filtered
~~~~

This should create a folder called filtered_data within which there would be different folders for the bio pages of faculties in different universities.

## The Strategy

The Automated Crawler works in conjunction with the BioPageClassifier (link).
It uses a webcrawler that crawls the webpages and downloads relevant faculty bio pages at regular intervals of time.

### Steps
1. Start crawling on a initial set of university webpages (this can be updated from wikipedia (link)).
2. For each university fetch the main page and corresponding links.
3. Exclude those link which may not direct to the faculty pages. This is done heuristically by using several university websites as reference.
4. For each link, fetch the content and classify wheter Bio page or not using BioPageClassifier. Save bio page.
5. Fetch links from this page and continue as step 2 recursively.

![scrapy](https://docs.scrapy.org/en/latest/_images/scrapy_architecture_02.png)

Note: The Breadth First Search approach is used to get relevant pages as soon as possible without going too deep into unfruitful links.
Note: The webcrawler is set to explore only links upto 5 levels from the main page. 

Both these are customizable based on settings in the webcrawler.

## The Tools
The webcrawler is created by subclassing a Scrapy spider. Then several customizations are added to aid additional functionality specific to Automated Crawler.

### Scrapy
Scrapy is a fast high-level web crawling and web scraping framework, used to crawl websites and extract structured data from their pages. It can be used for a wide range of purposes, from data mining to monitoring and automated testing.

### URL Tree
This is being used to get the website structure. It also allows to traverse a link only once and not repeat it.

## The Crawler

The key code within the webcrawler is the **parse** function whose relevant contents are given below:


#### Classify BioPage by passing the html content to the  Classifier
~~~~
isBio = self.bio_identifier.is_bio_html_content(response.xpath('//*').get())

if isBio:
    text = BeautifulSoup(response.xpath('//*').get(), features="html.parser").get_text()
    tokens = nltk.word_tokenize(text)
    normalized_text = ' '.join([word for word in tokens if word.isalnum()])
    normalized_text += '\n'+response.url

    hash_text = hashlib.md5(response.url.encode()) 
    file_name = hash_text.hexdigest()

    with open(folder_name+"/"+file_name+".txt", "w") as file:
        file.write(normalized_text)
        
~~~~

#### Extract all links uniquely and filter using exclude words
~~~~
AllLinks = LinkExtractor(allow_domains = self.allowed_domains[0], unique=True).extract_links(response)

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

~~~~
#### Only add in queue those links that are filtered, not traversed before
~~~~
            yield scrapy.Request(url=link.url, callback = self.parse)
~~~~

#### Save the URL Tree every 50 iterations
~~~~
if self.record[domain]%50 == 0:
   print('\n Crawled {} Bio-pages of {} University ...'.format(self.record[domain], domain.capitalize()))
   self.tree.save2file(folder_name+"/00__"+str(self.record[domain])+"_tree.txt")             
~~~~

## Some Harmless Errors/Warnings
You may face some warning or error messages in very rare cases. However, they won't interrupt the crawler execution flow. They occur because of the below mentioned reasons.
#### Reasons
1. File not found 404 error - Some links are bad and the crawler cannot reach the page.
2. Error when it tries to scrape the file objects embedded/linked to the web pages.

## Successful Execution

**Screenshot depicting the Execution log of Crawler:**
![cmd](https://github.com/chmvkalyan/ExpertSearchCrawler/blob/develop/images/Crawler_Execution_Log.jpg)


**Screenshot depicting the Extraction of Bio-pages to Output Folder:**
![output](https://github.com/chmvkalyan/ExpertSearchCrawler/blob/develop/images/Bio_Page_Extraction_To_OutputFolder.jpg)


**Screenshot depicting the Faculty Directory Page URL of Stanford University in URL Tree:**
![DirPage](https://github.com/chmvkalyan/ExpertSearchCrawler/blob/develop/images/Faculty_Directory_Page_on_URL_Tree.jpg)


**Screenshot depicting the Faculty Bio-page URLs of Stanford University in URL Tree:**
![BioPage](https://github.com/chmvkalyan/ExpertSearchCrawler/blob/develop/images/Faculty_Bio_Page_URLs_from_Stanford_University.jpg)


#### Link to the URL Tree and Faculty Bio-pages Crawled from Stanford University as a sample:
https://github.com/chmvkalyan/ExpertSearchCrawler/tree/develop/Crawled_Bio_Data


Please contact cheerla3@illinois.edu for live demo or in case of any question. Thank you!!
