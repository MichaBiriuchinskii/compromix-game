from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import spacy
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.decomposition import PCA
import pickle
import os.path
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Word Connections API")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load language models
logger.info("Loading language models...")
try:
    # Load English model
    en_model = spacy.load("en_core_web_md")
    logger.info("English model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading English model: {str(e)}")
    logger.info("Downloading English model...")
    spacy.cli.download("en_core_web_md")
    en_model = spacy.load("en_core_web_md")

try:
    # Load French model
    fr_model = spacy.load("fr_core_news_md")
    logger.info("French model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading French model: {str(e)}")
    logger.info("Downloading French model...")
    spacy.cli.download("fr_core_news_md")
    fr_model = spacy.load("fr_core_news_md")

# Cache for word vectors
vector_cache = {}

# Define request models
class WordRequest(BaseModel):
    word: str
    language: Optional[str] = "en"

class ScoreRequest(BaseModel):
    word: str
    word1: str
    word2: str
    language: Optional[str] = "en"

class WordPairRequest(BaseModel):
    word_pair: List[str]
    count: int = 5
    language: Optional[str] = "en"

# Common vocabulary words for each language (for suggestions)
common_words = {
    'en': [
        "nature", "landscape", "horizon", "valley", "climate", "environment",
        "terrain", "water", "earth", "view", "geography", "ecosystem",
        "beach", "forest", "lake", "river", "island", "cliff", "hill",
        "shore", "panorama", "wilderness", "wave", "summit", "canyon",
        "field", "garden", "park", "coast", "bay", "gulf", "sea", "plant",
        "animal", "weather", "storm", "rain", "wind", "sun", "cloud", "sky",
        "space", "planet", "star", "universe", "galaxy", "atmosphere",
        "travel", "journey", "adventure", "exploration", "discovery"
    ],
    'fr': [
        "nature", "paysage", "horizon", "vallée", "climat", "environnement",
        "terrain", "eau", "terre", "vue", "géographie", "écosystème",
        "plage", "forêt", "lac", "rivière", "île", "falaise", "colline",
        "rive", "panorama", "wilderness", "vague", "sommet", "canyon",
        "champ", "jardin", "parc", "côte", "baie", "golfe", "mer", "plante",
        "animal", "météo", "tempête", "pluie", "vent", "soleil", "nuage", "ciel",
        "espace", "planète", "étoile", "univers", "galaxie", "atmosphère",
        "voyage", "trajet", "aventure", "exploration", "découverte"
    ]
}

# Get the appropriate model for the language
def get_model(language):
    if language.lower() == "fr":
        return fr_model
    return en_model  # Default to English

# Get vector for a word
def get_word_vector(word, language="en"):
    """Get word vector with caching"""
    cache_key = f"{word}_{language}"
    if cache_key in vector_cache:
        return vector_cache[cache_key]
    
    model = get_model(language)
    doc = model(word)
    
    # Check if the word is recognized and has a vector
    if doc.has_vector and doc.vector_norm > 0:
        vector = doc.vector
        # Cache the vector
        vector_cache[cache_key] = vector
        return vector
    else:
        # Try without preprocessing
        vector = model.vocab.get_vector(word)
        vector_cache[cache_key] = vector
        return vector

# Check if a word is valid
def is_valid_word(word, language="en"):
    """Check if a word exists in the model's vocabulary"""
    model = get_model(language)
    
    # Try as a token
    doc = model(word)
    if len(doc) == 1 and not doc[0].is_punct and not doc[0].is_space:
        return True
    
    # Try directly in vocabulary
    if word in model.vocab:
        return True
    
    return False

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if vec1.shape[0] == 0 or vec2.shape[0] == 0:
        return 0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


# Add these improved scoring functions to your app.py

def normalize_vector(vector):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_midpoint(vec1, vec2):
    """Calculate the normalized midpoint between two vectors"""
    midpoint = (vec1 + vec2) / 2
    return normalize_vector(midpoint)

def dot_product(vec1, vec2):
    """Calculate dot product between two normalized vectors"""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return np.dot(vec1_norm, vec2_norm)

def calculate_rrf_score(word_vector, word1_vector, word2_vector):
    """Calculate Reciprocal Rank Fusion score"""
    # Calculate similarity to each anchor word
    sim1 = dot_product(word_vector, word1_vector)
    sim2 = dot_product(word_vector, word2_vector)
    
    # Convert similarities to ranks (higher similarity = lower rank number = better)
    # For simplicity, we'll convert directly since we don't have a full ranking
    rank1 = 1 / (1.001 - sim1)  # Prevent division by zero
    rank2 = 1 / (1.001 - sim2)  # Higher similarity gives lower rank number
    
    # Apply RRF formula: (a + b) / (a * b) with k=10
    k = 10
    a = rank1 + k
    b = rank2 + k
    rrf_score = (a + b) / (a * b)
    
    # Scale to 0-1 range for consistency with other scores
    # This scaling is approximate and may need adjustment
    return min(rrf_score * 5, 1.0)

# Cache for vocabulary and embeddings to avoid recomputation
vocabulary_cache = defaultdict(dict)

# Load a larger vocabulary for each language - these could be pickle files with word embeddings
# This is a simplified version - you would need to create these files
def load_vocabulary(language="en", max_words=50000):
    """Load or compute a large vocabulary with embeddings"""
    cache_key = f"{language}_{max_words}"
    
    if cache_key in vocabulary_cache:
        return vocabulary_cache[cache_key]
    
    # Path to cached vocabulary file (if it exists)
    cache_path = f"vocab_{language}_{max_words}.pkl"
    
    # Check if we have a cached version
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                vocabulary_cache[cache_key] = pickle.load(f)
                logger.info(f"Loaded vocabulary cache for {language}: {len(vocabulary_cache[cache_key])} words")
                return vocabulary_cache[cache_key]
        except Exception as e:
            logger.error(f"Error loading vocabulary cache: {e}")
    
    # If no cache, build vocabulary (this is slow and should be done offline)
    # For now, we'll use a smaller set of common words as a placeholder
    logger.info(f"Building vocabulary for {language}...")
    model = get_model(language)
    
    # Use model's lexicon as vocabulary source (this is spaCy-specific)
    # For a real implementation, you might want to load from a frequency list
    words = {}
    count = 0
    
    # Get words from model vocabulary
    for word in model.vocab.strings:
        # Skip non-word tokens and very short words
        if len(word) < 3 or not word.isalpha():
            continue
            
        # Get word vector if it exists
        if word in model.vocab and model.vocab[word].has_vector:
            words[word] = model.vocab[word].vector
            count += 1
            
            if count >= max_words:
                break
    
    # Store in cache
    vocabulary_cache[cache_key] = words
    logger.info(f"Built vocabulary with {len(words)} words")
    
    # Optionally save to disk for future use
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(words, f)
            logger.info(f"Saved vocabulary cache to {cache_path}")
    except Exception as e:
        logger.error(f"Error saving vocabulary cache: {e}")
    
    return words

# Request model for top words
class TopWordsRequest(BaseModel):
    word_pair: List[str]
    count: int = 1000
    language: Optional[str] = "en"
    include_pca: bool = True


# API endpoints
@app.get("/")
async def root():
    return {"message": "Word Connections API is running", "languages": ["en", "fr"]}

@app.post("/embedding")
async def get_embedding_endpoint(request: WordRequest):
    """Get the embedding vector for a word"""
    try:
        word_vector = get_word_vector(request.word, request.language)
        return {"word": request.word, "embedding": word_vector.tolist(), "language": request.language}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.post("/score")
async def calculate_score_endpoint(request: ScoreRequest):
    """Calculate similarity score between a word and the midpoint of two other words"""
    try:
        word = request.word.lower()
        word1 = request.word1.lower()
        word2 = request.word2.lower()
        language = request.language
        
        # Check if the word is valid
        valid_word = is_valid_word(word, language)
        
        # Get word vectors
        try:
            word_vector = get_word_vector(word, language)
            word1_vector = get_word_vector(word1, language)
            word2_vector = get_word_vector(word2, language)
            
            # Calculate normalized midpoint
            midpoint = calculate_midpoint(word1_vector, word2_vector)
            
            # Calculate dot product similarity to midpoint
            midpoint_similarity = dot_product(word_vector, midpoint)
            
            # Calculate RRF score (balance between both anchor words)
            rrf_score = calculate_rrf_score(word_vector, word1_vector, word2_vector)
            
            # Calculate distance from the "line" between the two words
            # (This identifies words that are directly between rather than off to the side)
            line_vector = normalize_vector(word2_vector - word1_vector)
            projection = np.dot(word_vector - word1_vector, line_vector)
            distance_from_line = np.linalg.norm(
                word_vector - (word1_vector + projection * line_vector)
            )
            line_score = max(0, 1 - (distance_from_line * 2))
            
            # Combine scores (weight can be adjusted based on preference)
            # Higher weight on midpoint_similarity rewards words closer to exact middle
            combined_score = (midpoint_similarity * 0.6) + (rrf_score * 0.3) + (line_score * 0.1)
            
            # Scale to 0-1 and clamp
            final_score = max(0, min(1, combined_score))
            
            return {
                "word": word,
                "word1": word1,
                "word2": word2,
                "score": float(final_score),
                "valid_word": valid_word,
                "language": language,
                "metrics": {
                    "midpoint_similarity": float(midpoint_similarity),
                    "rrf_score": float(rrf_score),
                    "line_score": float(line_score)
                }
            }
        except Exception as e:
            logger.error(f"Vector calculation error: {str(e)}")
            # Return a fallback score for unknown words
            return {
                "word": word,
                "word1": word1,
                "word2": word2,
                "score": 0.5,  # Neutral score
                "valid_word": valid_word,
                "language": language
            }
    except Exception as e:
        logger.error(f"Error calculating score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating score: {str(e)}")

@app.post("/suggest_words")
async def suggest_words(request: WordPairRequest):
    """Suggest words that are close to the midpoint of two given words"""
    try:
        word1, word2 = request.word_pair
        word1 = word1.lower()
        word2 = word2.lower()
        language = request.language
        
        # Get word vectors
        word1_vector = get_word_vector(word1, language)
        word2_vector = get_word_vector(word2, language)
        
        # Calculate midpoint
        midpoint = (word1_vector + word2_vector) / 2
        
        # Get vocabulary for the selected language
        vocabulary = common_words.get(language, common_words['en'])
        
        # Filter out the original words
        vocabulary = [w for w in vocabulary if w.lower() not in [word1, word2]]
        
        # Calculate scores
        scores = []
        for word in vocabulary:
            try:
                word_vector = get_word_vector(word, language)
                sim_score = cosine_similarity(word_vector, midpoint)
                score = (sim_score + 1) / 2
                score = max(0, min(1, score))  # Clamp between 0 and 1
                scores.append({"word": word, "score": float(score)})
            except Exception as e:
                logger.error(f"Error scoring word '{word}': {str(e)}")
                continue
        
        # Sort by score in descending order and take top 'count'
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        return {"suggestions": sorted_scores[:request.count], "language": language}
    
    except Exception as e:
        logger.error(f"Error suggesting words: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error suggesting words: {str(e)}")
    
@app.post("/top_midpoint_words")
async def get_top_midpoint_words(request: TopWordsRequest):
    """Get words closest to the midpoint between two words"""
    try:
        word1, word2 = request.word_pair
        word1 = word1.lower()
        word2 = word2.lower()
        language = request.language
        
        # Get word vectors
        word1_vector = get_word_vector(word1, language)
        word2_vector = get_word_vector(word2, language)
        
        # Calculate normalized midpoint
        midpoint = calculate_midpoint(word1_vector, word2_vector)
        
        # Load vocabulary (or smaller subset for testing)
        vocabulary = load_vocabulary(language, max_words=10000)  # Adjust size as needed
        
        # Calculate distances to midpoint
        word_distances = []
        for word, vector in vocabulary.items():
            # Skip the anchor words
            if word.lower() in [word1, word2]:
                continue
                
            # Calculate similarity to midpoint
            similarity = dot_product(normalize_vector(vector), midpoint)
            word_distances.append((word, similarity))
        
        # Sort by similarity (descending)
        word_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_words = word_distances[:request.count]
        
        # Optionally compute PCA if requested
        pca_data = None
        if request.include_pca:
            # Extract vectors for PCA
            vectors = [vocabulary[word] for word, _ in top_words[:100]]  # Use top 100 for PCA
            
            # Add anchor words and midpoint
            vectors.append(word1_vector)
            vectors.append(word2_vector)
            vectors.append(midpoint * np.linalg.norm(word1_vector))  # Scale midpoint to similar magnitude
            
            # Fit PCA
            pca = PCA(n_components=3)
            projected = pca.fit_transform(vectors)
            
            # Extract results
            pca_results = {
                "top_words": [{"word": word, "coords": coords.tolist()} 
                             for (word, _), coords in zip(top_words[:100], projected[:-3])],
                "anchor1": {"word": word1, "coords": projected[-3].tolist()},
                "anchor2": {"word": word2, "coords": projected[-2].tolist()},
                "midpoint": {"coords": projected[-1].tolist()},
                "explained_variance": pca.explained_variance_ratio_.tolist()
            }
            pca_data = pca_results
        
        return {
            "word_pair": [word1, word2],
            "top_words": [{"word": word, "score": float(score)} for word, score in top_words],
            "pca_data": pca_data
        }
    
    except Exception as e:
        logger.error(f"Error getting top midpoint words: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting top midpoint words: {str(e)}")


@app.get("/languages")
async def get_available_languages():
    """Get available languages"""
    return {
        "languages": ["en", "fr"],
        "word_counts": {
            "en": len(common_words['en']),
            "fr": len(common_words['fr'])
        }
    }

@app.get("/dictionary/check/{word}")
async def check_word(word: str, language: str = "en"):
    """Check if a word exists in the model's vocabulary"""
    valid = is_valid_word(word, language)
    
    return {
        "word": word,
        "valid": valid,
        "language": language
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)