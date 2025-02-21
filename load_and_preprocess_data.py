import pandas as pd


def load_and_preprocess_data(dataset_path: str) -> pd.DataFrame:
        """Load and clean IMDB dataset"""
        df = pd.read_csv(dataset_path)
        # Clean data
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce') 
        df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')#converting string to int
        df['Gross'] = df['Gross'].str.replace('[\$,]', '', regex=True).astype(float) #removing punctuation and converting string to float
        df['Genre'] = df['Genre'].apply(lambda x: [s.lower() for s in x.split(', ')] if isinstance(x, str) else []) #converting string to list of strings
        df['Director'] = df['Director'].str.lower()
        df["Series_Title"] = df["Series_Title"].str.lower()
        df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')#converting string to int
        df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')#converting string to number (float)
        df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float) #removing non-numerical chars and converting to a number
        cast_column = ['Star1', 'Star2', 'Star3', 'Star4']
        df['Cast'] = df[cast_column].values.tolist()#creating a list of cast memebers
        df = df.drop(columns=cast_column)#removing individual cast columns
        return df


dataset_path = 'imdb_top_1000.csv'        
df = load_and_preprocess_data(dataset_path)
