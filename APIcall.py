#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install requests


# In[3]:


import requests

url = "https://api.themoviedb.org/3/trending/movie/day?language=en-US"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ODM5NzkxMDM0NDI0ZDEwOGE4OTVjNDU2ZWZjMGMwMiIsInN1YiI6IjY1MDI3MDhmZGI0ZWQ2MTA0MzA5ZjAwYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.spH5JF6tvXkuCE8cb4HDuuzQdMfi3hhBqSmO3DSf3Zk"
}

response = requests.get(url, headers=headers)

print(response.text)


# In[4]:


import requests
import json

api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ODM5NzkxMDM0NDI0ZDEwOGE4OTVjNDU2ZWZjMGMwMiIsInN1YiI6IjY1MDI3MDhmZGI0ZWQ2MTA0MzA5ZjAwYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.spH5JF6tvXkuCE8cb4HDuuzQdMfi3hhBqSmO3DSf3Zk'

# Base URL for TMDb API
base_url = 'https://api.themoviedb.org/3/trending/movie/day'

# Set the initial page number and create an empty list for storing movie data
page_number = 1
all_trending_movies = []

# Set up headers with your API key
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

while True:
    # Set the parameters for the request, including the page number
    params = {
        'language': 'en-US',
        'page': page_number,
        'api_key': api_key
    }

    try:
        # Send a GET request to TMDb API
        response = requests.get(base_url, headers=headers, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract movie information from the response
            if data['results']:
                all_trending_movies.extend(data['results'])
                page_number += 1
            else:
                break  # No more movies found
        else:
            print('Error:', response.status_code)
            break  # Exit the loop on error
    except Exception as e:
        print('An error occurred:', str(e))
        break  # Exit the loop on error

# Save the collected data to a JSON file
output_filename = 'trending_movies.json'

with open(output_filename, 'w', encoding='utf-8') as json_file:
    json.dump(all_trending_movies, json_file, ensure_ascii=False, indent=4)

print(f'Trending movies data saved to {output_filename}')


# In[5]:


import pandas as pd
# Load the JSON data into a pandas DataFrame
df1 = pd.read_json(output_filename)

# Display the DataFrame
print(df1.head())


# In[6]:


df1.info()


# In[7]:


df1.drop(['backdrop_path', 'original_title', 'poster_path'], axis = 1, inplace = True)


# In[8]:


df1.head(10)


# In[9]:


df1[['adult']].value_counts()


# In[10]:


df1.drop(['adult'], axis = 1, inplace = True)


# In[11]:


df1[['video']].value_counts()


# In[12]:


df1.drop(['video'], axis = 1, inplace = True)


# In[13]:


df1[['media_type']].value_counts()


# In[14]:


df1.drop(['media_type'], axis = 1, inplace = True)


# In[15]:


import numpy as np
df1['year'] = pd.to_datetime(df1['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[16]:


df1.head(10)


# In[17]:


import requests
import json
import pandas as pd

api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ODM5NzkxMDM0NDI0ZDEwOGE4OTVjNDU2ZWZjMGMwMiIsInN1YiI6IjY1MDI3MDhmZGI0ZWQ2MTA0MzA5ZjAwYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.spH5JF6tvXkuCE8cb4HDuuzQdMfi3hhBqSmO3DSf3Zk'

# Base URLs for TMDb API
base_url = 'https://api.themoviedb.org/3/'
genres_url = f'{base_url}genre/movie/list'

# Set up headers with your API key
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Fetch genre data from TMDb API
try:
    response = requests.get(genres_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        genre_data = response.json()
        
        # Create a dictionary to map genre IDs to their names
        genre_dict = {genre['id']: genre['name'] for genre in genre_data['genres']}
    else:
        print('Error fetching genre data:', response.status_code)
        genre_dict = {}
except Exception as e:
    print('An error occurred while fetching genre data:', str(e))
    genre_dict = {}

# Function to map genre IDs to genre names
def map_genre_ids_to_names(genre_ids):
    if isinstance(genre_ids, list):
        return [genre_dict.get(genre_id, 'Unknown') for genre_id in genre_ids]
    else:
        return ['Unknown']

# Update the genre_ids column in your existing DataFrame 'df'
df1['genre_ids'] = df1['genre_ids'].apply(map_genre_ids_to_names)

# Rename the 'genre_ids' column to 'genres' if needed
df1.rename(columns={'genre_ids': 'genres'}, inplace=True)


# In[18]:


df1.head(10)


# In[19]:


df2 = pd.read_csv('movies_metadata.csv', low_memory=False)


# In[20]:


df2.head(10)


# In[21]:


df2.info()


# In[22]:


df2.drop(['adult', 'genres', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'poster_path',
         'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count'], axis = 1, inplace = True)


# In[23]:


df2['id'] = pd.to_numeric(df2['id'], errors='coerce').fillna(0).astype(int)


# In[24]:


df = pd.merge(df1, df2, on = 'id')


# In[25]:


df.head(10)


# In[26]:


df.info()


# In[27]:


df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df['year'].astype('int64')
df['budget'] = df['budget'].astype('float64')


# In[28]:


df.info()


# In[29]:


df.describe()


# In[30]:


df = df.copy().loc[df['vote_count'] >= 399]
df.shape


# In[31]:


df.isna().sum()


# In[32]:


C = df["vote_average"].mean()


# In[33]:


#function to calculate weighted rating of each movie
def weighted_rating(x, m=399, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[34]:


df['score'] = df.apply(weighted_rating, axis=1).round(2)
df.head(10)


# In[35]:


import ast  # Import the ast module for literal string parsing
# Define a function to extract the collection name
def extract_collection_name(collection):
    try:
        collection_dict = ast.literal_eval(collection)  # Parse the string as a dictionary
        return collection_dict.get('name', None)  # Get the 'name' key from the dictionary
    except (ValueError, SyntaxError):
        return np.nan  # Return None for non-parsable values or missing 'name'

# Apply the function to the column
df['belongs_to_collection'] = df['belongs_to_collection'].apply(extract_collection_name)


# In[36]:


df.head(10)


# In[37]:


# Define a function to extract and format the production companies as lists
def extract_and_format_companies(companies):
    try:
        company_list = ast.literal_eval(companies)  # Parse the string as a list of dictionaries
        company_names = [entry['name'] for entry in company_list]  # Extract the 'name' from each dictionary
        return company_names  # Return the names as a list
    except (ValueError, SyntaxError):
        return np.nan  # Return None for non-parsable values

# Apply the function to the column
df['production_companies'] = df['production_companies'].apply(extract_and_format_companies)


# In[38]:


# Define a function to extract and format the production countries as lists
def extract_and_format_countries(countries):
    try:
        country_list = ast.literal_eval(countries)  # Parse the string as a list of dictionaries
        country_names = [entry['name'] for entry in country_list]  # Extract the 'name' from each dictionary
        return country_names  # Return the names as a list
    except (ValueError, SyntaxError):
        return np.nan  # Return None for non-parsable values

# Apply the function to the column
df['production_countries'] = df['production_countries'].apply(extract_and_format_countries)


# In[39]:


df.head(10)


# In[40]:


(df['runtime']==0).sum()


# In[41]:


df['runtime'].replace(0, df['runtime'].mean(), inplace=True)
df['runtime'] = df['runtime'].round()


# In[42]:


(df['budget']==0).sum()


# In[43]:


#Right method?
df['budget'].replace(0, df['budget'].mean(), inplace=True)


# In[44]:


#Right method?
df['revenue'].replace(0, df['revenue'].mean(), inplace=True)


# In[45]:


df.head(10)


# In[46]:


import matplotlib.pyplot as plt


# In[47]:


# Create a list to hold all genres
all_genres = []

# Iterate through the DataFrame to collect all unique genres
for genre_list in df['genres']:
    all_genres.extend(genre_list)

# Count the occurrences of each genre
genre_counts = pd.Series(all_genres).value_counts()

# Create a bar plot to visualize the number of movies per genre
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar')
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[48]:


plt.figure(figsize=(10, 6))
plt.hist(df['year'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Movie Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.show()


# In[49]:


import seaborn as sns
sns.displot(data=df, x='year', kind='hist', bins = 20, kde=True,
            color='#fdc100', facecolor='#06837f', edgecolor='#64b6ac', line_kws={'lw': 3}, aspect = 1.85)
plt.title('Distribution of Movie Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')


# In[50]:


# Create a list to hold all genres
all_genres = []

# Iterate through the DataFrame to collect all genres
for genre_list in df['genres']:
    all_genres.extend(genre_list)

# Count the occurrences of each genre
genre_counts = pd.Series(all_genres).value_counts()

# Create a bar plot to visualize the top 10 movie genres
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Movies w.r.t. Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[97]:


from scipy.stats import linregress
plt.figure(figsize=(10, 6))
plt.scatter(df['budget'], df['revenue'], alpha=0.5, color='skyblue')
plt.title('Scatter Plot of Movie Budget vs. Revenue')
plt.xlabel('Budget')
plt.ylabel('Revenue')

# Calculate the linear regression line
slope, intercept, r_value, p_value, std_err = linregress(df['budget'], df['revenue'])
x_values = np.array(df['budget'])
y_values = intercept + slope * x_values

# Plot the regression line
plt.plot(x_values, y_values, color = 'black', label='Regression Line')

# Show the legend
plt.legend()

plt.show()


# In[52]:


# Create a list to hold all production companies
all_companies = []

# Iterate through the DataFrame to collect all production companies
for company_list in df['production_companies']:
    all_companies.extend(company_list)

# Count the occurrences of each production company
company_counts = pd.Series(all_companies).value_counts()

# Sort the production companies by the number of movies in descending order
company_counts = company_counts.sort_values(ascending=True).tail(10)

# Create a horizontal bar plot to visualize the top production companies
plt.figure(figsize=(12, 6))
company_counts.plot(kind='barh', color='skyblue')
plt.title('Top 10 Production Companies')
plt.xlabel('Number of Movies')
plt.ylabel('Production Company')
plt.tight_layout()

# Show the plot
plt.show()


# In[53]:


plt.figure(figsize=(10, 6))
plt.hist(df['runtime'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Number of Movies')
plt.show()


# In[54]:


# Create a list to hold all genres
all_genres = []

# Iterate through the DataFrame to collect all genres
for genre_list in df['genres']:
    all_genres.extend(genre_list)

# Count the occurrences of each genre
genre_counts = pd.Series(all_genres).value_counts().head(10)

# Create a new DataFrame for genre counts by year
genre_counts_by_year = df.apply(lambda row: pd.Series(row['genres']), axis=1)
genre_counts_by_year['year'] = df['year']

# Use pivot_table to get counts of each genre by year
genre_counts_by_year = genre_counts_by_year.melt(id_vars='year', value_name='Genre')
genre_counts_by_year = pd.pivot_table(genre_counts_by_year, 
                                      index='year', columns='Genre', 
                                      values='Genre', aggfunc='count', fill_value=0)

# Create a stacked bar plot to visualize genre counts over time
plt.figure(figsize=(12, 6))
genre_counts_by_year.plot(kind='bar', stacked=True, colormap='Paired')
plt.title('Stacked Bar Chart of Movie Genres Over Time')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()


# In[55]:


import seaborn as sns

plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[57]:


pip install wordcloud


# In[111]:


from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.title('The Most Common Word in Movie Overviews\n', fontsize=20, weight=600, color='#333d29')
wc = WordCloud(max_words=1000, min_font_size=10,
                height=800,width=1600,background_color="white").generate(' '.join(df['overview']))

plt.imshow(wc)


# In[110]:


from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.title('The Most Common Word in Movie Titles\n', fontsize=20, weight=600, color='#333d29')
wc = WordCloud(max_words=1000, min_font_size=10,
                height=800,width=1600,background_color="white").generate(' '.join(df['title']))

plt.imshow(wc)


# In[60]:


pop= df.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# In[64]:


# Sort the DataFrame by 'score' in descending order and select the top ten movies
top_ten_movies = df.sort_values(by='score', ascending=False).head(10)

# Create a bar chart to visualize the top ten movies by votes
plt.figure(figsize=(12, 6))
plt.barh(top_ten_movies['title'], top_ten_movies['score'])
plt.title('Top Ten Movies by Weighted Rating')
plt.xlabel('Weighted Rating')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()  # Invert the y-axis to show the highest vote on top
plt.tight_layout()

# Show the plot
plt.show()


# In[65]:


# Sort the DataFrame by 'vote_average' in descending order and select the top ten movies
top_ten_movies = df.sort_values(by='vote_average', ascending=False).head(10)

# Create a bar chart to visualize the top ten movies by votes
plt.figure(figsize=(12, 6))
plt.barh(top_ten_movies['title'], top_ten_movies['vote_average'], color='skyblue')
plt.title('Top Ten Movies by Votes')
plt.xlabel('Average Vote')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()  # Invert the y-axis to show the highest vote on top
plt.tight_layout()

# Show the plot
plt.show()


# In[109]:


#Most revenue w.r.t. collection
# Create an empty dictionary to store total revenue per production company
company_revenue = {}

# Iterate through the DataFrame to calculate total revenue per company
for index, row in df.iterrows():
    companies = row['production_companies']
    revenue = row['revenue']
    for company in companies:
        if company in company_revenue:
            company_revenue[company] += revenue
        else:
            company_revenue[company] = revenue

# Sort the dictionary by total revenue in descending order and select the top 5 companies
sorted_companies = dict(sorted(company_revenue.items(), key=lambda item: item[1], reverse=True))
top_5_companies = list(sorted_companies.keys())[:5]
top_5_revenues = list(sorted_companies.values())[:5]

# Create a bar chart to visualize the top 5 production companies by revenue
plt.figure(figsize=(6, 6))
plt.bar(top_5_companies, top_5_revenues, width = 0.4)
plt.title('Top 5 Production Companies by Revenue')
plt.xlabel('Production Company')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[76]:


# Group the data by 'release_year' and calculate the average revenue per year
average_revenue_by_year = df.groupby('year')['revenue'].mean().reset_index()

# Create a line plot to visualize average movie revenue over time
plt.figure(figsize=(10, 5))
plt.plot(average_revenue_by_year['year'], average_revenue_by_year['revenue'], marker='o', linestyle='-', color='skyblue')
plt.title('Average Movie Revenue Over Time (Grouped by Year)')
plt.xlabel('Release Year')
plt.ylabel('Average Revenue')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# In[80]:


# Define the bins (e.g., decades)
bins = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Bin the 'release_year' column and create a new column 'decade'
df['decade'] = pd.cut(df['year'], bins=bins, labels=['1930-1940', '1940-1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020'])

# Group the data by 'decade' and calculate the average revenue per decade
average_revenue_by_decade = df.groupby('decade')['revenue'].mean().reset_index()

# Create a line plot to visualize average movie revenue over decades
plt.figure(figsize=(10, 5))
plt.plot(average_revenue_by_decade['decade'], average_revenue_by_decade['revenue'], marker='o', linestyle='-', color='skyblue')
plt.title('Average Movie Revenue Over Decades')
plt.xlabel('Decade')
plt.ylabel('Average Revenue')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# In[83]:


# Define the bins (e.g., decades)
bins = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Bin the 'release_year' column and create a new column 'decade'
df['decade'] = pd.cut(df['year'], bins=bins, labels=['1930-1940', '1940-1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020'])

# Group the data by 'decade' and calculate the average rating per decade
average_rating_by_decade = df.groupby('decade')['score'].mean().reset_index()

# Create a line plot to visualize average movie rating over decades
plt.figure(figsize=(10, 5))
plt.plot(average_rating_by_decade['decade'], average_rating_by_decade['score'], marker='o', linestyle='-', color='skyblue')
plt.title('Average Movie Rating Over Decades')
plt.xlabel('Decade')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# In[91]:


# Flatten the 'genres' column to separate individual genres
flat_df = df.explode('genres')

# Group the data by genre and calculate the count of movies per genre
genre_counts = flat_df.groupby('genres').size().reset_index(name='count')

# Sort the genres by count in descending order and select the top 10 genres
top_10_genres = genre_counts.nlargest(9, 'count')

# Calculate the total count of genres not in the top 10 and group them as 'Other'
other_count = genre_counts['count'].sum() - top_10_genres['count'].sum()
top_10_genres = top_10_genres.append({'genres': 'Other', 'count': other_count}, ignore_index=True)

# Create a pie chart to visualize the distribution of movie genres
plt.figure(figsize=(6, 6))
plt.pie(top_10_genres['count'], labels=top_10_genres['genres'], autopct='%1.1f%%')
plt.title('Pie Chart - Top 10 Movie Genres')
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
plt.tight_layout()

# Show the plot
plt.show()


# In[101]:


# Flatten the 'genres' column to separate individual genres
flat_df = df.explode('genres')

# Filter rows with the 'Drama' genre
drama_df = flat_df[flat_df['genres'] == 'Drama']

# Sort the DataFrame by 'score' in descending order
sorted_drama_df = drama_df.sort_values(by='score', ascending=False)

# Select the top 10 rated movies
top_10_drama_movies = sorted_drama_df.head(10)

# Create a bar plot to visualize the top 10 rated drama movies with ratings displayed
plt.figure(figsize=(10, 6))
bars = plt.barh(top_10_drama_movies['title'], top_10_drama_movies['score'], color='skyblue')
plt.title('Top 10 Rated Drama Movies')
plt.xlabel('Weighted Rating')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()  # Invert the y-axis to display the highest rating at the top

# Add ratings as text labels to the bars
for bar, rating in zip(bars, top_10_drama_movies['score']):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f'{rating:.1f}', ha='center', va='center')

plt.tight_layout()

# Show the plot
plt.show()


# In[108]:


# Group the data by 'original_language' and count the number of movies for each language
language_counts = df['original_language'].value_counts().reset_index()
language_counts.columns = ['original_language', 'count']

# Select the top 5 languages
top_5_languages = language_counts.head(5)

# Calculate the total count of languages not in the top 5 and group them as 'Other'
other_count = language_counts['count'].sum() - top_5_languages['count'].sum()
top_5_languages = top_5_languages.append({'original_language': 'Other', 'count': other_count}, ignore_index=True)

# Create a bar plot to visualize the number of movies per original language (top 5 + 'Other')
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5_languages['original_language'], top_5_languages['count'], color = ['thistle', 'lightcoral', 'peachpuff', 'paleturquoise', 'lightgray', 'darkseagreen'])
plt.title('Number of Movies by Original Language')
plt.xlabel('Original Language')
plt.ylabel('Number of Movies')

# Add movie counts as text labels on each bar
for bar, count in zip(bars, top_5_languages['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(count), ha='center', va='bottom')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




