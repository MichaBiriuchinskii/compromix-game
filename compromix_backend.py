from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import spacy
from typing import List, Dict, Any, Tuple, Optional

# Define request models
class WordRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class ScoreRequest(BaseModel):
    word: str
    word1: str
    word2: str

class GameRequest(BaseModel):
    word1: str
    word2: str
    n: int = 10

app = FastAPI()

# Enable CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main game class
class CompromixGame:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_name=None, vocab_file=None):
        if cls._instance is None:
            cls._instance = cls(model_name, vocab_file)
        return cls._instance
    
    def __init__(self, model_name=None, vocab_file=None):
        """
        Initialize the game with a spaCy model for French.
        
        Args:
            model_name: Name of the spaCy model to load
            vocab_file: File containing a custom French vocabulary
        """
        self.model = self._load_model(model_name or "fr_core_news_md")
        self.target_word = None
        self.valid_words = self._get_valid_words(vocab_file, max_words=20000)
        print(f"Vocabulaire chargé: {len(self.valid_words)} mots")
        
    def _load_model(self, model_name):
        """Load the spaCy model for French"""
        try:
            print(f"Chargement du modèle spaCy {model_name}...")
            return spacy.load(model_name)
        except OSError:
            print(f"Modèle {model_name} non trouvé. Téléchargement en cours...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)
    
    def _get_valid_words(self, vocab_file=None, max_words=20000):
        """
        Get a list of valid words for the game.
        """
        # If a vocabulary file is provided, use it
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
            return words
        
        # Use words from the spaCy model's vocabulary
        print("Extraction des mots du vocabulaire spaCy...")
        words = []
        for word in self.model.vocab:
            # Only include words with vectors and of reasonable length
            if word.has_vector and len(word.text) > 2 and word.is_alpha and not word.is_stop:
                words.append(word.text.lower())
            
            if len(words) >= max_words:
                break
        
        # If we have too few words, add some common French words
        if len(words) < 500:
            common_french_words = [
                "maison", "voiture", "chat", "chien", "livre", "table", "arbre", "fleur",
                "ordinateur", "téléphone", "jardin", "cuisine", "école", "travail", "famille",
                "ami", "restaurant", "café", "ville", "village", "pays", "montagne", "mer",
                "soleil", "lune", "étoile", "route", "rue", "rivière", "forêt", "magasin",
                "marché", "hôpital", "médecin", "patient", "maladie", "santé", "sport", 
                "musique", "film", "télévision", "radio", "internet", "voyage", "vacances",
                "été", "hiver", "printemps", "automne", "jour", "nuit", "matin", "soir"
                # Add more common words as needed
            ]
            
            for word in common_french_words:
                if word not in words:
                    words.append(word)
        
        # Save the vocabulary for future use
        with open("french_vocab_spacy.txt", "w", encoding="utf-8") as f:
            for word in words:
                f.write(word + "\n")
        print("Vocabulaire sauvegardé dans french_vocab_spacy.txt")
        
        return words
    
    def get_vector(self, word):
        """Get the vector for a word"""
        return self.model(word.lower()).vector
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        vec1_reshaped = vec1.reshape(1, -1)
        vec2_reshaped = vec2.reshape(1, -1)
        return float(cosine_similarity(vec1_reshaped, vec2_reshaped)[0][0])
    
    def find_intermediate_words(self, word1, word2, n=10):
        """
        Find n words that are intermediate between word1 and word2 in the vector space.
        
        Returns:
            List of tuples (word, similarity_to_midpoint, position, ideal_percentage)
        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        # Calculate the exact midpoint
        mid_vec = (vec1 + vec2) / 2
        
        # Calculate similarities for each word in the vocabulary
        results = []
        for word in self.valid_words:
            if word != word1 and word != word2:
                word_vec = self.get_vector(word)
                
                # Similarity to the midpoint
                sim_to_midpoint = float(cosine_similarity(
                    word_vec.reshape(1, -1), mid_vec.reshape(1, -1))[0][0])
                
                # Relative position on the word1-word2 axis
                position = self.calculate_position_between(word, word1, word2)
                
                # Calculate ideal percentage (100% = exactly in the middle)
                ideal_percentage = (1 - abs(position - 0.5) * 2) * 100
                
                results.append((word, sim_to_midpoint, position, ideal_percentage))
        
        # Sort by similarity to the midpoint
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
    
    def calculate_position_between(self, word, word1, word2):
        """
        Calculate where a word is positioned between two others on a scale from 0 (word1) to 1 (word2).
        """
        vec = self.get_vector(word)
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        # Direction vector from word1 to word2
        direction = vec2 - vec1
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-10:
            return 0.5  # The two words are identical
        
        # Project the word onto the word1-word2 axis
        vec_rel = vec - vec1
        projection = np.dot(vec_rel, direction) / direction_norm**2
        
        # Limit between 0 and 1
        return max(0, min(1, projection))
    
    def start_new_game(self, word=None):
        """Start a new game with a random or specified target word"""
        if word:
            self.target_word = word
        else:
            self.target_word = random.choice(self.valid_words)
        return self.target_word
    
    def guess(self, word):
        """
        Evaluate a user's guess.
        
        Returns:
            float: Similarity score between 0 and 1
            bool: True if it's the target word
        """
        if not self.target_word:
            raise ValueError("No active game. Use start_new_game() first.")
            
        similarity = self.similarity(word, self.target_word)
        return similarity, word == self.target_word

# Initialize the game on API startup
game = None

@app.on_event("startup")
async def startup_event():
    global game
    # Look for an existing vocabulary file first
    vocab_file = "french_vocab_spacy.txt" if os.path.exists("french_vocab_spacy.txt") else None
    game = CompromixGame.get_instance(vocab_file=vocab_file)
    print("Modèle chargé et prêt")

# API routes
@app.post("/embedding")
async def get_embedding(request: WordRequest):
    """
    Return the embedding vector for a word.
    """
    try:
        embedding = game.get_vector(request.word).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul de l'embedding: {str(e)}")

@app.post("/similarity")
async def get_similarity(request: SimilarityRequest):
    """
    Calculate the direct similarity between two words.
    """
    try:
        similarity = game.similarity(request.word1, request.word2)
        return {"similarity": float(similarity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul de la similarité: {str(e)}")

@app.post("/score")
async def calculate_score(request: ScoreRequest):
    """
    Calculate the similarity score between a word and the midpoint between two other words.
    """
    try:
        word = request.word
        word1 = request.word1
        word2 = request.word2
        
        # Calculate relative position on the word1-word2 axis
        position = game.calculate_position_between(word, word1, word2)
        
        # Calculate similarity to the midpoint
        vec = game.get_vector(word)
        vec1 = game.get_vector(word1)
        vec2 = game.get_vector(word2)
        midpoint = (vec1 + vec2) / 2
        
        similarity = cosine_similarity(vec.reshape(1, -1), midpoint.reshape(1, -1))[0][0]
        
        # Calculate ideal percentage (100% = exactly in the middle)
        ideal_percentage = (1 - abs(position - 0.5) * 2) * 100
        
        return {
            "score": float(similarity),
            "position": float(position),
            "ideal_percentage": float(ideal_percentage)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul du score: {str(e)}")

@app.post("/intermediate_words")
async def find_intermediate_words(request: GameRequest):
    """
    Find words that are intermediate between two given words.
    """
    try:
        word1 = request.word1
        word2 = request.word2
        n = request.n
        
        intermediate = game.find_intermediate_words(word1, word2, n)
        
        # Format the response
        result = []
        for word, sim, position, ideal_pct in intermediate:
            result.append({
                "word": word,
                "similarity": float(sim),
                "position": float(position),
                "ideal_percentage": float(ideal_pct)
            })
        
        return {"intermediate_words": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche de mots intermédiaires: {str(e)}")

# Add a root route to provide basic information
@app.get("/")
async def root():
    return {
        "name": "Compromix Game API",
        "description": "API pour un jeu d'énigmes sémantiques explorant les mots proches dans l'espace sémantique",
        "endpoints": [
            {"method": "POST", "path": "/embedding", "description": "Obtenir le vecteur d'embedding d'un mot"},
            {"method": "POST", "path": "/similarity", "description": "Calculer la similarité entre deux mots"},
            {"method": "POST", "path": "/score", "description": "Calculer le score d'un mot entre deux autres mots"},
            {"method": "POST", "path": "/intermediate_words", "description": "Trouver des mots intermédiaires entre deux mots"}
        ]
    }

# Entry point to run the application
if __name__ == "__main__":
    # If run as a script, launch the API
    uvicorn.run(app, host="0.0.0.0", port=8000)