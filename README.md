## Persona Predicta Project
### Creating user personas from survey data


### Team
Kevin Eliasen, Fred Lambuth, Sean Oslin and Dom Pedrotti


### Project sponsor
Rosenfeld Media  https://rosenfeldmedia.com


### Request from Rosenfeld Media (RM)
We were asked to apply data science methods to quantitative and qualitative survey data collected by RM, a New York conference-producing and publishing company.


### Goal
In order to save time and labor, RM wanted to automate the process of extracting actionable information from future surveys. In particular, the company wanted the ability to create survey-respondent profiles that can inform conference design and assist with conference marketing.


### Included in the Github for this project
- README with background on the project
- Original data in a single Excel spreadsheet
- Survey tool
- Jupyter notebook with Python functions and findings
- Text files for each stage of the data science pipeline, including functions
-Background information on the survey, resources on personas


### Background (from an interview with Lou Rosenfeld, President, Rosenfeld Media)
"RM wanted to add a new conference to their portfolio. An opportunity seemed to arise in the area of research.

"Seeds for the survey and conference were planted in late 2018 months ago. As a publisher and conference producer, RM has noticed a change in the area of research for products and services, with multiple trends in the research world coming to a head. 

"Trends seen
- Research is moving away from silos and focusing on collaboration across job functions. This trend seems to be happening organically, not necessarily being required by management.
- Increased emphasis on managing data across teams to find insights.
- Increased focus on moving research operations from small projects to projects that can scale. This is coinciding with enabling changes to infrastructure and management operational skills.
- Research functions have been added to organizations where none existed before.
- Research has raised its profile. Researchers may now have a seat at the table, but not necessarily know what to do there.
- The number of industry researchers has increased.
- More non-researchers are doing research. Some may be involved in research but not even know it.

"These trends are forcing researchers to work in new ways, in new settings and with new types of professions. Additionally, research is concerned with scaling findings and encompassing subject matters that have not been seen before. 

"RM decided to create new ways for people to come together to discuss how research is changing. The survey was the first step in determining the desire of people."


### Survey design
RM is experienced with on-line survey design and implementation. RM piloted the survey with a dozen people. They accidentally omitted survey restrictions prohibiting responders from skipping required questions and submitting incomplete surveys. RM consciously omitted many questions regarding respondent demographics and location as not relevant to the goal of the survey.


### Survey layout
The online survey consists of 23 main questions, divided among free-text, Likert-scale and combined-response questions. Six questions were multipart (all Likert). In total, the survey consistent of 68 questions. The survey tool is included in this GitHub. While the survey tool included numerous required questions, respondents could submit an incomplete survey.


### Survey output
The survey response far exceeded expectations: 100 anticipated and 860 actual responses. Approximately 150 incomplete surveys were submitted, creating challenges for analysis.


### Previous analysis
RM requested UX experts to tabulate th

https://public.tableau.com/profile/sean.oslin#!/vizhome/WhoDoesResearchandHowDoTheyLearn_15693334108800/StoryDraft1

### Acquire


### Wrangle


### Explore


### Model


### Minimum viable project (MVP)
The MVP used a support vector machine (SVM) classification model with the attributes taxo_id, the max of years_of_exp (all 10 questions) and company_size to predict the 5 previously determined personas. We used the same attributes as the previously the analysis 

The model replicated the results of the previous analysis

https://public.tableau.com/profile/sean.oslin#!/vizhome/WhoDoesResearchandHowDoTheyLearn_15693334108800/StoryDraft1
