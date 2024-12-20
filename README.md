# **End of a Movie: Predictable or Not?**  

### Abstract

This project explores the extent to which a movie’s ending can be predicted using various factors from our dataset, such as genre, actors, directors, time period, and other metrics like ratings and box office performance. We aim to analyze patterns and correlations that may reveal underlying trends in movie endings and examine how societal and historical shifts might influence these trends. Additionally, we will explore variations in endings across different countries, hypothesizing that cultural factors play a role in shaping audience expectations, consequently, the types of endings filmmakers choose. Ultimately, this project seeks to answer: can we foresee a film's ending before watching it? This exploration offers valuable insights into audience preferences, cultural influences, and the changing dynamics of the film industry.

### Research Questions

To guide our analysis and improve our ability to predict movie endings, we will address the following questions:
     
1. How does the genre of a movie influence the type of ending it has (happy, tragic, neutral)?
Certain genres may have a higher likelihood of happy or tragic endings. This question will help us understand how genre conventions shape audience expectations.
       
2. How have happy and tragic endings evolved over time?
By analyzing movie endings across different time periods, we aim to identify historical trends and examine how societal shifts might influence the tone of a film’s conclusion.
       
3. What role do key personnel (actors and directors) play in shaping a movie's ending?
Do certain actors or directors have a preference for particular types of endings, and do their choices influence the overall predictability of a movie’s outcome?
       
4. Is there a correlation between a movie’s ending and its success (ratings, box office revenue, etc.)?
We will explore whether happy or tragic endings have any impact on a movie's popularity or financial performance.
     
5. Do cultural differences influence the type of movie ending? Are there notable differences in the types of endings preferred or produced in different countries, suggesting cultural factors at play?
    
6. Do budget and production scale affect the type of ending chosen?
Exploring whether high-budget films tend to favor certain endings (e.g., happy endings for wider audience appeal) could reveal if financial considerations impact storytelling choices.

7. How do movie endings vary by film length ?
This question would explore whether movies of different lengths (e.g., short vs. feature-length) tend to have different types of endings.

8. Do sequels or franchise movies follow different patterns in their endings compared to standalone films?
Investigating whether franchise movies are more likely to have open-ended or happy endings to encourage future installments could reveal a unique trend in storytelling within cinematic universes.
       


### Additional Datasets 

We employ an additional dataset The Movie Database (TMDb) to enhance our analysis and provide a more comprehensive understanding of movie endings. This dataset contains information on movies, including runtime, ratings, revenue, crew members, and actors. We will merge this dataset with the data we have to enrich our analysis and explore additional factors that may influence movie endings.

We use the [TMDb API](https://developer.themoviedb.org/docs/getting-started) to access the dataset and retrieve relevant information. After cleaning and preprocessing the data, we will merge it with our existing dataset to create a more comprehensive dataset for analysis.

We utilized the **ICBe dataset**, a comprehensive resource on international crises, sourced from the [ICBe GitHub Repository](https://github.com/CenterForPeaceAndSecurityStudies/ICBEdataset). This dataset provides detailed information on crisis events, their associated actors, and key metadata, making it invaluable for analyzing the interplay between historical events and movies.

From this dataset, we incorporated two main files:
- **`ICBe_V1.1_events_agreed.Rds`**: Includes detailed information on event actors, timelines, and conditions, such as interactions, fatalities, military forces, and cooperative or antagonistic actions.
- **`ICBe_V1.1_crises_markdown.Rds`**: Contains crisis-level summaries, with a unique crisis identifier (`crisno`), geographic regions, actors involved, and narrative overviews.

These files were merged to create a unified dataset that captures:
- **Key Attributes**: Crisis ID, Crisis Title, Start Year, End Year, Event Type, Fatalities, and Actors (Countries).
- **Temporal Coverage**: 450 crisis events spanning from 1918 to 2015, aligning well with the movie release years in our dataset.

This unified dataset allows us to robustly connect movies to historical events. For example:
- We minimized the number of movies without matching events. Even with a threshold of **200 related events per movie**, the dataset still includes over **12,000 movies**, ensuring a wealth of data for meaningful analysis.



<!-- If you plan to use additional datasets, list them here, along with your approach to acquiring, managing, processing, and enriching them. Demonstrate that you have reviewed the relevant documentation and examples, and have a realistic expectation of what the data entails (considering factors like size and format). -->

Here’s the text translated into English and written in the past tense:  

---

### **Methods**

To answer our research questions and evaluate the predictability of movie endings, we employed a combination of data processing, statistical analysis, and machine learning techniques. Our methods included the following steps:  

**Data Preprocessing and Cleaning:**  
We began by cleaning the dataset to handle missing values, standardize categorical fields (e.g., genre, country), and normalize numerical data (e.g., ratings, box office revenues).  

**Exploratory Data Analysis (EDA):**  
We conducted EDA to identify initial patterns and trends. Visualization techniques were used to understand distributions and relationships, such as the prevalence of happy vs. tragic endings across genres, time periods, and countries. We also examined the evolution of movie endings over time, highlighting shifts or patterns that aligned with societal changes or cinematic trends.  

**Statistical Analysis:**  
To evaluate relationships between movie characteristics and ending types, we applied statistical tests. For example, chi-square tests determined if certain genres were more likely to have specific endings, and correlation analysis assessed the relationship between ending types and success metrics (e.g., ratings).  
 


## Contributions of Team Members

 Below is a detailed breakdown of roles and achievements:



### 1. Xiangyu  
- Responsible for analyzing and addressing research question 3 and 4.  
 
### 2. Mylène  
- Conducted in-depth analysis of research questions 6, 7, and 8.  
- Developed the structure of the website.  
- Ensured consistent styling throughout the website to maintain a uniform appearance.
- Updated the README

### 3. Nicolas  
- Developed and implemented the solution for research question 2. 
- Assisted in ensuring consistent styling across the website. 

### 4. Marie  
- Analyzed and documented the results for research question 1.  

### 5. Charles  
- Conducted the analysis and implementation for research question 4. 


# How to run

## Quickstart

     # clone project
     git clone <project link>
     cd <project repo>

     # install requirements
     pip install -r pip_requirements.txt
     How to use the library

**Dowload the CMU Movies Dataset and extract the folder in the 'Data' folder.**


# Project Structure:

     ├── Data                      <- Project data files
     │   ├── ICBe   
     │   ├── MovieSummaries  
     ├── layouts 
     ├── assets 
     │   ├── css   
     │   ├── img  
     │   ├── js  
     ├── sections                  <- html file for each page 
     ├── src                       <- Source code
     │   ├── data                  <- created and uploaded data   
     │   ├── scripts               <- scripts for research questions 
     │
     ├── index.html                <- website's mainpage
     ├── results.ipynb             <- a well-structured notebook showing the results
     ├── scripts.js                <- scripts
     ├── style.css                 <- website's style
     ├── .gitignore                <- List of files ignored by git
     ├── pip_requirements.txt      <- File for installing python dependencies
     └── README.md
