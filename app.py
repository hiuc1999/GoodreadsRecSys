import pandas as pd
import numpy as np
import _pickle as cpickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st 


st.set_page_config(page_title="Book Recommender", page_icon=":books:", layout="wide")
st.markdown("""<style> #MainMenu, header, footer {visibility: hidden;} </style>""",unsafe_allow_html=True)
st.title("Get Book Recommendations")
st.write('This web app can give book recommendation by a hybrid recommendation model built from  UCSD Goodreads dataset.')
st.write('The hybird recommendation model combines content-based and collaboration filtering, supporting item-to-item and user-to-item recommendation. ')

# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    


df_books = cpickle.load(open('df_books_authors.pickle','rb'))
booklist= cpickle.load(open('book_list.pickle','rb'))


books_latent = cpickle.load(open('latent_matrix_books.pickle','rb'))
rating_latent = cpickle.load(open('latent_matrix_rating.pickle','rb'))


def user_to_user(seeduserID):
    seed_user = np.array(latent_rating_u2u.loc[seeduserID]).reshape(1,-1)
    similarities = cosine_similarity(latent_rating_u2u, seed_user, dense_output=True)
    index = latent_rating_u2u.index.tolist()
    similarities = pd.DataFrame(similarities, index = index)
    similarities.columns = ['similarity_score']
    similarities.sort_values('similarity_score', ascending=False, inplace=True)
    similarities = similarities.iloc[1:]
    similarities = similarities[similarities['similarity_score'] > 0]
    
    return similarities[:min(len(similarities),30)]

def user_to_book(userID):
    global rating_latent, books_latent, latent_rating_u2u
    latent_rating_u2u = rating_latent.transpose()

    sim_user = user_to_user(userID)

    book_read = latent_rating_u2u.loc[userID]
    book_read = book_read.loc[(book_read > 0)]
    book_read = book_read.sort_values(ascending=False)[:min(30,len(book_read))]

    sim = latent_rating_u2u[latent_rating_u2u.index.isin(sim_user.index)]
    sim = sim.loc[:, (sim > 0).any(axis=0)].loc[~sim.index.isin(book_read.index)]
    avgrating = sim.replace(0,np.nan).apply(np.nanmean).dropna()
    avgrating = avgrating.sort_values(ascending=False).index[:min(50,len(avgrating))]

    similarities = cosine_similarity(books_latent[df_books.index[df_books['work_id'].isin(list(avgrating))]],books_latent[df_books.index[df_books['work_id'].isin(list(book_read.index))]])
    cos = pd.DataFrame(similarities, index = list(avgrating), columns = list(book_read.index))
    rec = df_books[df_books['work_id'].isin(cos.index[[a for (a,b) in [divmod(i, 30) for i in np.argsort(similarities, axis=None)[-10:]]]][::-1])][['title','authors','isbn','description','average_rating']]

    return rec.iloc[:5]


def book_to_books(seedbookID, latentmatrix, rec_mode):
    if rec_mode == 'collaborative':
        seed_book = np.array(latentmatrix.loc[seedbookID]).reshape(1,-1)
    if rec_mode == 'content':
        seed_book = latentmatrix[df_books.index[df_books['work_id'] == seedbookID]]

    similarities = cosine_similarity(latentmatrix, seed_book, dense_output=True)

    if rec_mode == 'collaborative':
        index = latentmatrix.index.tolist()
    if rec_mode == 'content':
        index = df_books['work_id'].tolist()

    similarities = pd.DataFrame(similarities, index = index)
    similarities.columns = ['similarity_score']
    similarities.sort_values('similarity_score', ascending=False, inplace=True)
    similarities = similarities.iloc[1:]
    similarities = similarities[similarities['similarity_score'] > 0]

    return similarities

def similarity_scores(collaborative_score, content_score):
  #average both similarity scores
  df_sim = pd.merge(collaborative_score, pd.DataFrame(content_score['similarity_score']), left_index=True, right_index=True)
  df_sim['similarity_score'] = (df_sim['similarity_score_x'] + (df_sim['similarity_score_y'])*0.5)/2
  df_sim.drop("similarity_score_x", axis=1, inplace=True)
  df_sim.drop("similarity_score_y", axis=1, inplace=True)

  #sort by average similarity score
  df_sim.sort_values('similarity_score', ascending=False, inplace=True)

  #round similarity score
  df_sim['similarity_score'] = df_sim['similarity_score'].round(4)

  return  df_sim.head(20)

def get_recommendation(seed_book):
    global rating_latent, books_latent
    collaborative = book_to_books(seed_book, rating_latent, 'collaborative')
    content = book_to_books(seed_book, books_latent, 'content')
    rec = similarity_scores(collaborative, content)
    rec = pd.merge(df_books, rec, how='right', left_on='work_id', right_index=True).reset_index().drop(['index','similarity_score'], axis=1)
    rec = rec[['title','authors','isbn','description','average_rating']]
    return rec.iloc[:5]

def fetch_poster(isbn):
    url = "https://covers.openlibrary.org/b/isbn/{}-L.jpg".format(isbn)
    return url

tab1, tab2 = st.tabs(['Book-to-book', 'User-to-book'])

with tab1:
    st.header('Book-to-book Recommender')
    
    book_title = st.selectbox('Please choose the book title', booklist['title'].sort_values())
    
    if st.button('Get Recommendations!',key=1):
        bookid = booklist[booklist["title"] == str(book_title)]["work_id"].values[0]
        recommendation = get_recommendation(bookid)
        titles = recommendation['title'].tolist()
        isbn = recommendation['isbn'].tolist()
        authors = recommendation['authors'].tolist()
        descriptions = recommendation['description'].tolist()
        
    
        st.subheader(titles[0])
        st.image(fetch_poster(isbn[0]))
        st.write('ISBN: ',isbn[0])
        st.write('Author(s): ', ', '.join(authors[0]))
        st.write('Description: ')
        st.write(str(descriptions[0]))
        
        st.subheader(titles[1])
        st.image(fetch_poster(isbn[1]))
        st.write('ISBN: ',isbn[1])
        st.write('Author(s): ', ', '.join(authors[1]))
        st.write('Description: ')
        st.write(str(descriptions[1]))
        
        st.subheader(titles[2])
        st.image(fetch_poster(isbn[2]))
        st.write('ISBN: ',isbn[2], ', '.join(authors[2]))
        st.write('Description: ')
        st.write(str(descriptions[2]))
        
        st.subheader(titles[3])
        st.image(fetch_poster(isbn[3]))
        st.write('ISBN: ',isbn[3])
        st.write('Author(s): ', ', '.join(authors[3]))
        st.write('Description: ')
        st.write(str(descriptions[3]))
        
        st.subheader(titles[4])
        st.image(fetch_poster(isbn[4]))
        st.write('ISBN: ',isbn[4])
        st.write('Author(s): ',', '.join(authors[4]))
        st.write('Description: ')
        st.write(str(descriptions[4]))

    
        
    

with tab2:
    st.header('User-to-book Recommender')
    userID = st.number_input('Please enter UserID (from 0 to 18891)', min_value=0, max_value=18891)
    
    if st.button('Get Recommendations!',key=2):
        recommendation = user_to_book(userID)
        titles = recommendation['title'].tolist()
        isbn = recommendation['isbn'].tolist()
        authors = recommendation['authors'].tolist()
        descriptions = recommendation['description'].tolist()
        
        st.subheader(titles[0])
        st.image(fetch_poster(isbn[0]))
        st.write('ISBN: ',isbn[0])
        st.write('Author(s): ', ', '.join(authors[0]))
        st.write('Description: ')
        st.write(str(descriptions[0]))
        
        st.subheader(titles[1])
        st.image(fetch_poster(isbn[1]))
        st.write('ISBN: ',isbn[1])
        st.write('Author(s): ', ', '.join(authors[1]))
        st.write('Description: ')
        st.write(str(descriptions[1]))
        
        st.subheader(titles[2])
        st.image(fetch_poster(isbn[2]))
        st.write('ISBN: ',isbn[2], ', '.join(authors[2]))
        st.write('Description: ')
        st.write(str(descriptions[2]))
        
        st.subheader(titles[3])
        st.image(fetch_poster(isbn[3]))
        st.write('ISBN: ',isbn[3])
        st.write('Author(s): ', ', '.join(authors[3]))
        st.write('Description: ')
        st.write(str(descriptions[3]))
        
        st.subheader(titles[4])
        st.image(fetch_poster(isbn[4]))
        st.write('ISBN: ',isbn[4])
        st.write('Author(s): ',', '.join(authors[4]))
        st.write('Description: ')
        st.write(str(descriptions[4]))

        



    