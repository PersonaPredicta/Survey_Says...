## Survey Sayz... Project
### Creating user personas from survey data


### Team
Kevin Eliasen, Fred Lambuth, Sean Oslin and Dom Pedrotti


### Project sponsor
Rosenfeld Media (RM)  https://rosenfeldmedia.com


### Request from Rosenfeld Media (RM)
We were asked to apply data science methods to quantitative and qualitative survey data collected by RM, a New York conference-producing and publishing company.


### Goal
To save time and labor, RM wants to automate the process of extracting actionable information from future surveys. In particular, the company wants the ability to create survey-respondent profiles that can inform conference design and assist with conference marketing.


### Included in the Github for this project   
- README with background on the project
- Original data in a single Excel spreadsheet
- Survey tool
- Text files for each stage of the data science pipeline, including functions
- Background information on the survey, resources on building personas
- Analysis from the first team that used different analysis techniques

https://github.com/PersonaPredicta


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
RM is experienced with on-line survey design and implementation. RM piloted the survey with a dozen people. When setting up the survey, RM accidentally omitted restrictions prohibiting responders from skipping required questions and submitting incomplete surveys. RM consciously excluded many questions regarding respondent demographics and location as not relevant to the goal of the survey.


### Survey layout
The online survey consists of 23 main questions, divided among free-text, Likert-scale and combined-response questions. Six questions were multipart (all Likert). In total, the survey consistent of 68 questions. The survey tool is included in the GitHub for this project. While the survey tool included numerous required questions, respondents could submit an incomplete survey.


### Survey distribution
The survey period was in August and September, 2019. RM promoted the survey heavily in RM social media (Twitter and LinkedIn). They sent a link to to the RM mailing list. They marketed it through UX communities managed by RM. RM gave advanced access on tickets to the conference for people who completed the survey. A RM e-book incentive was offered to the first 250 people who finished the survey. RM contacted people outside the UX community and asked to send the survey to their communities (e.g. business anthropology, market research, usability engineering, and information architecture).


### Survey output
The survey response far exceeded expectations: 100 anticipated and 860 actual responses. Approximately 150 incomplete surveys were submitted. The data was output to a single Excel spreadsheet.


### Previous analysis
RM requested UX experts to clean and tabulate the survey data. An initial analysis generated five personas with quantitative and qualitative data describing 5 personas. The overall analysis is included in the GitHub folder for this project. Visualized analysis of just the quantitative data can be found at the following link:

https://public.tableau.com/profile/sean.oslin#!/vizhome/WhoDoesResearchandHowDoTheyLearn_15693334108800/StoryDraft1

### Acquire
RM made a few changes to the data before giving it to our team. The most important change was reducing the 28 unique job titles of survey respondents to 9 'taxo_id' titles. These changes were applied by grouping like titles (e.g UX designer and UX researcher) and combining titles with only a small number of respondents (e.g. enthnologist) into the 'specialist' taxo_id. 

RM broke down the preferred conference size mixed-response question to include additional categories ('no preference', 'it depends', or 'virtual'). Our team accepted all of these changes to the raw data.  

### Wrangle
Data preparation included the following
- Converting all text answers to the Likert question (e.g. always, somewhat, never) to numbers
- For the mixed questions, the preselected choices (e.g. Yes, I was taught research at school; No, I was not taught research at school) were tabulated. The free text was read by the team and the responses, where possible, were rolled into the preselected choices (e.g. on-the-job learning was categorized as not learning at school). For the preferred conference format mixed-response question, the free-text responses that could not be assigned to one of the existing choices did fall into two distinct new catergories - 'mixed' and 'no preference' - and were coded accordingly. Next we created three new non-exclusive, binary variables: single-track, multi-track and unconference. If a respondent specified one or more one of these three options, a 1 was placed in the corresponding variable(s). Respondents who stated no preference left as 0.
- The number of missing values for required questions was significant, with the number of missing values increasing as the survey progressed. We decided to eliminate all responses that did not complete the last set of required Likert questions. The total number of records dropped from 868 to 726. No additional deletions were necessary.


### Hypotheses
#### Minimum viable product (MVP)
H<sub>0</sub> : The data analysis that lead to the intial persona IDs and descriptive statistics CANNOT be reproduced using data science techniques.

H<sub>a</sub> : Intial persona IDs and descriptive statistics CAN be reproduced using data science techniques. 

##### Final product
H<sub>0</sub> : A re-classification of personas will result in substantially the same results as the initial personas.

H<sub>a</sub> : Re-classified personas will create substantially different results as the initial personas.

### Explore
We attempted to predict each individual respondent's likelyhood to attend the conference score and then identify key features driving the target. We discovered that the questions about likelyhood to attend were the only ones to have any predictive power.  Because of this finding, we abandoned this line of research. 

Replace did any part of your education include research with [none, a little, some, a lot, and gobs*] Leikerts indicating how much learning about research came from various sources: [formal education, self-taught, side classes, or on-the-job]
Add "no preference" and "it depends" to preferred conference size OR change the existing options to Leikerts
Change the existing options for preferred conference format to Leikerts OR add "no preference" and "mixed".



### Model


### Minimum viable project (MVP)
The MVP used a support vector machine (SVM) classification model with the attributes taxo_id, the max of years_of_exp (all 10 questions) and company_size to predict the 5 previously determined personas. We used the same attributes as the previously the analysis 

The model replicated the results of the previous analysis


