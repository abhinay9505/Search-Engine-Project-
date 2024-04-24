import streamlit as st
st.set_page_config(
    page_title="",
    page_icon="🏏🏏",
)
st.title("Search Engine : A Subtitle viewer of Movies and web series")

User_query=st.text_area("Subtitle to search")

if st.button("Search_top_movies"):

    st.title("Top Subtitled Movies")
        
    import pandas as pd
    df = pd.read_csv(r"C:\Users\Abinay Rachakonda\Desktop\sreach_Engine\datasets\Sub_Titles_BERT.csv")
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')
    def clean_content(text):
        text = re.sub(timestamp_pattern, '', text)  # Remove timestamps
        text = re.sub(unwanted_pattern, '', text)    # Remove unwanted characters
        text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
        text = text.strip()                          # Strip leading/trailing whitespace
        text = re.sub(r'^\d+', '', text.strip())     # Remove leading '1'
        return text.strip()
    def preprocess(raw_text):
        # Removing special characters and digits
        sentence = re.sub("[^a-zA-Z]", " ", raw_text)

        #change sentence to lower case
        sentence = sentence.lower()

        # tokenize into words
        tokens =nltk.word_tokenize(sentence)

        #Lemmatize
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        #remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]

        #Join and return
        return " ".join(filtered_tokens)
    from tqdm import tqdm
    tqdm.pandas()
    ## step3: Data Preprocessing on train data
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    # Regular expression pattern to remove timestamps
    timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
    # Regular expression pattern to remove unwanted characters like '\r', '\n', 'ï»¿'
    unwanted_pattern = r'[\r\nï»¿]'

    import re
    User_query = pd.Series(User_query)
    User_query.progress_apply(lambda x:  clean_content(x)) 

    User_query.progress_apply(lambda x:  preprocess(x)) 

    Query = User_query.progress_apply(model.encode)

    df['doc_vector_pretrained_bert'] = df['doc_vector_pretrained_bert'].str.replace('[', '').str.replace(']', '')

    # Assuming df is your DataFrame and the column name is 'doc_vector_pretrained_bert'
    df['doc_vector_pretrained_bert'] = df['doc_vector_pretrained_bert'].progress_apply(lambda x: [float(i) for i in x.split()])
    cos_sim = util.cos_sim(Query, df['doc_vector_pretrained_bert'])
    a=cos_sim.argsort()[:][::].tolist()
    single_list = []
    for sublist in a:
        single_list.extend(sublist)

    top_sub = df.iloc[sublist[::-1]]

    k=10
    st.write(top_sub[:k])


    st.title("Top Subtitles Generate")
    top_sub_top = top_sub.reset_index()
    st.write(top_sub_top.loc[0,'file_content_chunks'])