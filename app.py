import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("---")
st.markdown("### Enter the movie genre or feature you're looking for, and we'll recommend the best movies for you!")

load_dotenv()

@st.cache_resource
def initialize_chromadb():
    """Initialize ChromaDB connection and cache it"""
    try:
        chroma_client = chromadb.PersistentClient("./embeddingsDB")
        collection = chroma_client.get_or_create_collection(
            name="movie_embeddings",
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-3-small"
            )
        )
        return collection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def search_movies(query, n_results, min_rating=None):
    """Movie search function"""
    collection = initialize_chromadb()
    if collection is None:
        return None
    
    try:
        where_filter = {}
        if min_rating and min_rating > 0:
            where_filter = {"rating": {"$gte": min_rating}}
        
        response = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return response
    except Exception as e:
        st.error(f"Error during search: {e}")
        return None

def display_movie_results(response):
    """Display movie results"""
    if not response or not response['ids'][0]:
        st.warning("No movies found matching your search criteria.")
        return
    
    movies = response['ids'][0]
    documents = response['documents'][0]
    metadatas = response['metadatas'][0]
    distances = response['distances'][0]
    
    st.success(f"🎯 {len(movies)} movies found!")
    
    for i, (movie, doc, meta, distance) in enumerate(zip(movies, documents, metadatas, distances)):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### 🎬 {movie}")
                st.markdown(f"**Genre:** {meta.get('genre', 'Not specified')}")
                st.markdown(f"**Description:** {doc}")
                
            with col2:
                rating = meta.get('rating', 'N/A')
                similarity = max(0, (1 - distance) * 100)
                
                st.metric("⭐ Rating", f"{rating}")
                st.metric("🎯 Similarity", f"{similarity:.1f}%")
            
            st.markdown("---")

def main():
    with st.sidebar:
        st.header("🔍 Search Settings")
        
        n_results = st.slider(
            "Number of movies to show",
            min_value=1,
            max_value=8,
            value=3,
            help="You can get up to 8 movie recommendations"
        )
        
        min_rating = st.slider(
            "Minimum IMDB rating",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Movies below this rating will not be shown"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Search Tips")
        st.markdown("""
        **Try these search examples:**
        - "action-packed superhero movies"
        - "romantic comedy with happy ending"
        - "mind-bending science fiction"
        - "psychological horror thriller"
        - "emotional family drama"
        - "fast-paced crime thriller"
        - "fantasy adventure with magic"
        - "dark mystery with plot twists"
        """)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Movie search",
            placeholder="e.g., action-packed, thrilling movies",
            help="Enter the movie genre, feature, or topic you're looking for"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button or query:
        if query.strip():
            with st.spinner("Searching for movies..."):
                response = search_movies(
                    query=query.strip(),
                    n_results=n_results,
                    min_rating=min_rating if min_rating > 0 else None
                )
                
                if response:
                    display_movie_results(response)
        else:
            st.warning("Please enter a movie genre or feature to search.")
    
    st.markdown("---")
    st.markdown(
        "💡 **Note:** This system uses ChromaDB and OpenAI embeddings to make movie recommendations.",
        help="The system finds the most similar movies by comparing your input text with movie descriptions."
    )

if __name__ == "__main__":
    main()