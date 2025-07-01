# Movie Recommendation System
This project is a movie recommendation system using a vector database and OpenAI API.

## Dataset
[TV & Movie Metadata with Genres and Ratings (2023)](https://www.kaggle.com/datasets/gayu14/tv-and-movie-metadata-with-genres-and-ratings-imbd/data) dataset utilized in this project.

## Vector Database
ChromaDB is utilized in this project for storing and querying movie embeddings.

## Features
- **Semantic Search**: Find movies based on natural language descriptions
- **Rating Filters**: Filter movies by minimum IMDB rating
- **Customizable Results**: Get 1-8 movie recommendations

## Technologies Used
- **ChromaDB**: Vector database for storing movie embeddings
- **OpenAI API**: Text embeddings using `text-embedding-3-small` model
- **Streamlit**: Web application framework for the user interface

## Setup Instructions

### Installation
1. Clone the repository:
```bash
git clone https://github.com/firatyll/movie-recommendation.git
cd chromadb-vector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=Your Openai Api Key Here
DATASET_PATH=Your Dataset Path
```

4. Create the embeddings database directory in your project folder:
```bash
mkdir embeddingsDB
```

5. Load the dataset and create embeddings:
```bash
python data_pipeline.py
```

6. Run the Streamlit application:
```bash
streamlit run app.py
```

## How It Works

1. **Data Preprocessing**: Movie descriptions are cleaned and prepared from the dataset
2. **Batch Processing**: Movies are processed in batches of 100 to optimize API usage, prevent rate limiting, and avoid ChromaDB token limits per request
3. **Embedding Generation**: OpenAI's text-embedding-3-small model creates vector representations for each movie description
4. **Vector Storage**: ChromaDB stores embeddings with metadata (genre, rating) in persistent storage
5. **Semantic Search**: User queries are converted to embeddings and matched against stored vectors using cosine similarity
6. **Result Ranking**: Movies are ranked by similarity score and filtered by rating criteria

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
