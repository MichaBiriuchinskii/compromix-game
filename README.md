# Word Connections Game

Compromix is a strategic word game where players propose terms to gradually converge on a middle ground between two opposing concepts, aiming to find the perfect "compromise" word.
It would challenge players' sense of nuance, negotiation, and semantic balance, blending logic with creativity.

### Compromix: Examples of Opposing Concepts and Middle-Ground Words

1. **War** ↔ **Peace**  
   → *Middle Ground:* **Truce**  
   _(temporary cessation, not full peace nor ongoing war)_

2. **Hot** ↔ **Cold**  
   → *Middle Ground:* **Warm** or **Lukewarm**  
   _(a literal midpoint in temperature)_

3. **Freedom** ↔ **Control**  
   → *Middle Ground:* **Regulation**  
   _(structured freedom within boundaries)_

4. **Rich** ↔ **Poor**  
   → *Middle Ground:* **Middle-class** or **Modest**  
   _(neither extreme wealth nor poverty)_

5. **Optimism** ↔ **Pessimism**  
   → *Middle Ground:* **Realism**  
   _(accepting both positive and negative possibilities)_




## Setup for Local Development

### 1. Set Up the Python Backend

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python app.py
   ```

   The API will be available at http://localhost:8000

5. You can test the API is working by visiting http://localhost:8000 in your browser. You should see a JSON response: `{"message":"Word Connections API is running"}`

### 2. Set Up the Frontend

1. Simply open `index.html` in your web browser.
   
   - For the best experience, use a local server:
     ```bash
     # If you have Python installed
     python -m http.server 8080
     ```
     Then visit http://localhost:8080 in your browser.

2. The frontend is configured to connect to the local API by default:
   ```javascript
   // In index.html
   config.useLocalApi = true;
   ```

## Deployment

### 1. Deploy the Python Backend to PythonAnywhere

1. Create a PythonAnywhere account at https://www.pythonanywhere.com

2. Create a new web app:
   - Choose "Web" tab → "Add a new web app"
   - Select "Manual configuration" and Python 3.9

3. Set up your virtual environment:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.9 wordgame
   pip install -r requirements.txt
   ```

4. Configure the WSGI file:
   - Go to the "Web" tab and click on the WSGI configuration file link
   - Replace the content with:

   ```python
   import sys
   import os

   # Add your project directory to the sys.path
   path = '/home/yourusername/word-connections-game'
   if path not in sys.path:
       sys.path.insert(0, path)

   # Set environment variables
   os.environ['PYTHONPATH'] = path

   # Import your app from app.py
   from app import app as application
   
   # CORS setup (add this if needed)
   from fastapi.middleware.cors import CORSMiddleware
   application.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourusername.github.io"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

5. Update the "Working directory" in the "Web" tab to your project directory

6. Reload your web app from the PythonAnywhere dashboard

### 2. Deploy the Frontend to GitHub Pages

1. Create a new GitHub repository

2. Update the API configuration in `index.html`:
   ```javascript
   config.api.production = 'https://yourusername.pythonanywhere.com';
   config.useLocalApi = false;  // Change this to use production API
   ```

3. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/word-connections-game.git
   git push -u origin main
   ```

4. Enable GitHub Pages:
   - Go to your repository on GitHub
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: main → /(root) → Save

5. Your game will be available at https://yourusername.github.io/word-connections-game/

## Extending the Game

### Add More Word Pairs

Edit the `wordPairs` array in the frontend code:

```javascript
wordPairs: [
    { word1: 'mountain', word2: 'ocean' },
    { word1: 'fire', word2: 'water' },
    // Add more pairs here
]
```

### Use Different Embedding Models

You can change the model in the Python backend:

```python
# app.py
model = SentenceTransformer('all-MiniLM-L6-v2')
```

Replace with other models from the sentence-transformers library, like:
- 'paraphrase-MiniLM-L6-v2'
- 'all-mpnet-base-v2'

### Add Daily Challenges

Create a JSON file with daily challenges and update the frontend to fetch it.

## Troubleshooting

### API Connection Issues

- Check that the backend server is running
- Ensure CORS is properly configured
- Check the API URL in the frontend configuration

### Slow Embedding Calculation

- Consider using a smaller model
- Implement caching for common words
- Pre-compute embeddings for planned word pairs
