# Persona Predicta Project
## Creating user personas from survey data


## Team
Kevin Eliasen, Fred Lambuth, Sean Oslin and Dom Pedrotti


## Project sponsor
Rosenfeld Media (RM)  https://rosenfeldmedia.com


## Request from Rosenfeld Media (RM)
We were asked to apply data science methods to quantitative and qualitative survey data collected by RM, a New York conference producing and publishing company.


## Goal
In order to save time and labor, RM wanted to automate the process of extracting actionable information from future surveys. In particular, the company wants the ability to create survey-respondent profiles that can inform conference design and assist with conference marketing.


### Included in the Github for this project
- README with background on the project
- Original data in a single Excel spreadsheet
- Survey tool
- Python modules containing the functions used in Juypter Notebooks
- Jupyter notebook demonstrating analysis process
- Text files for each stage of the data science pipeline, including functions
- Background information on the survey, resources on building personas
- Analysis from a different team that used different analysis techniques


## Background (from an interview with Lou Rosenfeld, President, Rosenfeld Media)
"RM wanted to add a new conference to their portfolio. An opportunity seemed to arise in the area of research."

"Seeds for the survey and conference were planted in late 2018, months ago. As a publisher and conference producer, RM has noticed a change in the area of research for products and services, with multiple trends in the research world coming to a head." 

Trends seen:
- Research is moving away from silos and focusing on collaboration across job functions. This trend seems to be happening organically, not necessarily being required by management.
- Increased emphasis on managing data across teams to find insights.
- Increased focus on moving research operations from small projects to projects that can scale. This is coinciding with enabling changes to infrastructure and management operational skills.
- Research functions have been added to organizations where none existed before.
- Research has raised its profile. Researchers may now have a seat at the table, but not necessarily know what to do there.
- The number of industry researchers has increased.
- More non-researchers are doing research. Some may be involved in research but not even know it.

"These trends are forcing researchers to work in new ways, in new settings and with new types of professions. Additionally, research is concerned with scaling findings and encompassing subject matters that have not been seen before."

"RM decided to create new ways for people to come together to discuss how research is changing. The survey was the first step in determining the desire of people."


## Survey design
RM is experienced with on-line survey design and implementation. RM piloted the survey with a dozen people. When setting up the survey, RM accidentally omitted restrictions prohibiting responders from skipping required questions and submitting incomplete surveys. RM consciously excluded many questions regarding respondent demographics and location as not relevant to the goal of the survey.


## Survey layout
The online survey consists of 23 main questions, divided among free-text, Likert-scale and combined-response questions. Six questions were multipart (all Likert). In total, the survey consistent of 68 questions. The survey tool is included in the repository for this project. While the survey tool included numerous required questions, respondents could submit an incomplete survey.


## Survey output
The survey response far exceeded expectations: 100 anticipated and 860 actual responses. Approximately 150 incomplete surveys were submitted. The data was output to a single Excel spreadsheet.


## Previous analysis
RM requested UX experts to clean and tabulate the survey data. The initial statistical analysis can be found here: 

https://public.tableau.com/profile/sean.oslin#!/vizhome/WhoDoesResearchandHowDoTheyLearn_15693334108800/StoryDraft1

## Acquire
RM made a few changes to the data before giving it to our team. The most important change was reducing the 28 unique job titles of survey respondents to 9 'taxo_id' titles. These changes were applied by grouping like titles (e.g UX designer and UX researcher) and combining titles with only a small number of respondents (e.g. enthnologist) into the 'specialist' taxo_id. RM broke down the preferred conference size mixed-response question to include additional categories ('no preference', 'it depends', or 'virtual'). Our team accepted all of these changes to the raw data.  

## Wrangle
Data preparation included the following
- Encoding all text answers to the Likert question (e.g. always, somewhat, never) to numerical scores.
- For the mixed questions, the preselected choices (e.g. Yes, I was taught research at school; No, I was not taught research at school) were tabulated. The free text was read by the team and the responses, where possible, were rolled into the preselected choices (e.g. on-the-job learning was categorized as not learning at school). For the preferred conference format mixed-response question, the free-text responses that could not be assigned to one of the existing choices did fall into two distinct new catergories - 'mixed' and 'no preference' - and were coded accordingly. Next we created three new non-exclusive, binary variables: single-track, multi-track and unconference. If a respondent specified one or more one of these three options, a 1 was placed in the corresponding variable(s). Respondents who stated no preference left as 0.
- The number of missing values for required questions was significant, with the number of missing values increasing as the survey progressed. We decided to eliminate all responses that did not complete the last set of required Likert questions. The total number of records dropped from 868 to 726. No additional deletions were necessary.

## Explore

Started fitting a model to predict respondent's likely hood score. This was to test the hypothesis: If we can build a model that predicts likely hood, then we can identify key features that drive the target. After test of hypothesis, I discovered that the only features with high correlation to our target had any predictive model so the plan was scraped.

Replace did any part of your education include research with [none, a little, some, a lot, and gobs*] Leikerts indicating how much learning about research came from various sources: [formal education, self-taught, side classes, or on-the-job]
Add "no preference" and "it depends" to preferred conference size OR change the existing options to Leikerts
Change the existing options for preferred conference format to Leikerts OR add "no preference" and "mixed".

## Model

### Qualitative Data
#### Cleaning/Preparing
Each qualitative question's responses were prepared for analysis by modifying words down to the their common root (e.g. running -> run, heads -> head), punctuation was removed, as well as words that are unlikely to be useful for analysis (e.g. I, you, is, their). 
#### Vectorizing
After preperations, an scikit-learn algorithm scans the corpus of responses, creating a list of all words used. Each individual response per question is then given a frequency count for each word it contains. The great majority of word counts are at 0. The words to be counted can be manually tuned to ignore based on the minimum and maximum amount of responses that contain the word.
#### Topic Clustering
The word count matrix made up of all the reponses word frequency counts are fed to a Latent Dirichtlet Algorithm that iteratively discovers commonalities between responses based on their word appearances, and the frequency of the appearances. The algorithm will try to discover a user-specified amount of clusters. These word grouping clusters are the topics extracted from the responses.
#### Topic Discovery
Using visual and numerical scores to measure the validity of clusters, the parameters for deciding what words to be used in the word count alogrithm and the LDA algorithm were adjusted until distinct and interpretable topics could be discerned.
#### Topic Features
Each response were assigned a unique amount of topics, with each topic having a nominal label for human readability. These topics were added to the dataset they were generated from as a topic label assigned to each qualitative question, allowing for exploration for the trends and distributions among those answers.



### Minimum viable project (MVP)
The MVP used a support vector machine (SVM) classification model with the attributes taxo_id, the max of years_of_exp (all 10 questions) and company_size to predict the 5 previously determined personas. We used the same attributes as the previously the analysis 

The model replicated the results of the previous analysis


