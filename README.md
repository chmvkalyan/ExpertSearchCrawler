# ExpertSearchCrawler

The purpose of this code is to improve upon the data collection for the ExpertSearch project (link). 

## Improvement
In the ExpertSearch project the database is collected manually as follows:

- Go to university's website
- Go to department's webpage
- Go to faculty directory page
- Extract links of faculty bio pages
- Process and download from faculty bio page

This process is manual, strenous, time-consuming and not dynamic.

The proposed method in the ExpertSearchCrawler is to automate many of these tasks.

## The Strategy

The ExpertSearchCrawler works in conjunction with the BioPageClassifier (link).
It uses a webcrawler that crawls the webpages and downloads relevant faculty bio pages at regular intervals of time.

### Steps
- 1. Start crawling on a initial set of university webpages (this can be updated from wikipedia (link))
- 2. For each university fetch the main page and corresponding links
- 3. Exclude those link which may not direct to the faculty pages. This is done heuristically by using several university websites as reference.
- 4. For each link, fetch the content and classify wheter Bio page or not using BioPageClassifier. Save bio page
- 5. Fetch links from this page and continue as step 2 recursively

Note: The Breadth First Search approach is used to get relevant pages asap without going too deep into unfruitful links.
Note: The webcrawler is set to explore only links upto 5 levels from the main page

Both these are customizable based on settings in the webcrawler

