import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash
from script import CLIPModelHelper, ChromaDBManager
import os
import pandas as pd
import sqlite3
import zipfile
from PIL import Image
from zipfile import ZipFile

# Initialize Flask app
app = Flask(__name__)

# Set device and CLIP model ID
device = 'cpu'
model_ID = 'openai/clip-vit-base-patch32'
# model_ID='openai/clip-vit-large-patch14'

# Initialize CLIP model helper
clip_model_helper = CLIPModelHelper(model_ID, device)

# Set up ChromaDBManager
chroma_db_host = os.getenv("CHROMADB_HOST", "localhost")
chroma_db_manager = ChromaDBManager(host=chroma_db_host, port=8000)

# Initialize and create a collection
chroma_db_manager.initialize_collection('Lattt')

# Set Flask session configuration and secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'your_secret_key'

# Create a temporary directory for CSV and images
import tempfile
upload_dir = tempfile.mkdtemp()
os.makedirs(upload_dir, exist_ok=True)

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('user_db.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

# Define the home route
@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    else:
        return render_template('login.html')  # Render login page directly

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            conn = sqlite3.connect('user_db.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username=?', (username,))
            user = cursor.fetchone()

            if user and check_password_hash(user[2], password):
                session['username'] = username
                return redirect(url_for('index'))
            else:
                flash("Invalid login credentials.")
        except Exception as e:
            print(f"Error during login: {e}")
        finally:
            conn.close()
    elif request.method == 'GET':
        return render_template('login.html')

# Define the signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        try:
            conn = sqlite3.connect('user_db.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
        except Exception as e:
            print(f"Error inserting new user: {e}")
        finally:
            conn.close()

        flash("Signup successful! Please log in.")
        return redirect(url_for('login'))
    else:
        return render_template('signup.html')

# Define the logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Define the add route for adding data
@app.route('/add', methods=['POST', 'GET'])
def add():
    if request.method == 'GET':
        return render_template('add.html')
    elif request.method == 'POST':
        if 'csvFile' not in request.files or 'zipFile' not in request.files:
            return jsonify({'message': 'No file part'})
        csv_file = request.files['csvFile']
        zip_file = request.files['zipFile']

        # Save CSV file
        csv_file_path = os.path.join('static', 'data.csv')
        csv_file.save(csv_file_path)

        # Extract images from the ZIP file
        images_folder_path = 'static/images_save'
        zip_filename_without_extension = os.path.splitext(zip_file.filename)[0]
        images_folder_pathh = os.path.join('static/images_save', zip_filename_without_extension)

        with ZipFile(zip_file, 'r') as zip_ref:
            print('a')
            zip_ref.extractall(images_folder_path)
        # images_folder_pathh = os.path.join('static/images_save', zip_file.filename)
        # Read CSV file into a DataFrame
        data = pd.read_csv(csv_file_path)

        image_data = []
        metadatas = []
        df = []

        for index, row in data.iterrows():
            item_id = str(row['collection_changed'] + str(row['Serial No.']))

            metadata = {
                'gsm': row['gsm'],
                'Color': row['Color'],
                'Pattern': row['Pattern'],
                'End use': row['End use'],
                'Transparency': row['Transparency'],
                'composition': row['composition'],
                'collection': row['collection_changed'],
                'fabric type': row['fabric type'],
                'documents': str(row['collection_changed'] + str(row['Serial No.']))
            }
            filename = str(row['collection_changed'] + str(row['Serial No.'])) + '.jpg'  # Assuming the file format
            image_exact_path = os.path.join(images_folder_pathh, filename)

            # Open the image file
            try:
                img = Image.open(image_exact_path)
            except FileNotFoundError:
                print(f"File not found: {image_exact_path}")
                continue

            img = img.convert('RGB')
            img = img.resize((256, 256))
            img_array = np.array(img)
            image_data.append(img_array)
            metadatas.append(metadata)
            df.append(item_id)

        images_df = pd.DataFrame({'image': image_data})
        if not df:
            return jsonify({'message': len(data)}), 400

        images_df = clip_model_helper.get_all_images_embedding(images_df, 'image', device)
        images_df['flattened_embeddings'] = images_df['img_embeddings'].apply(lambda x: x.flatten().tolist())
        chroma_db_manager.add_data_to_collection(images_df, df, metadatas, 'flattened_embeddings', 'id_column')

    return jsonify({'message': 'Data added successfully'})

# Function to extract images from a ZIP file
def extract_zip(zip_file):
    images_folder_path = 'resized_images'  # Assuming the folder for resized images
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(images_folder_path)
    return images_folder_path

# Define the index route for querying data
@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        n_results = 10
        image_data = []  # Initialize an empty list to hold data about the images
        image_file_url = []

        if 'file' in request.files:
            image_file = request.files['file']
            if image_file.filename != '':
                filename = image_file.filename  # Secure the filename
                filename_with_extension = filename.rsplit('.', 1)[0] + '.jpg'  # Ensure the filename has '.jpg' extension
                image_path = os.path.join('static/img', filename_with_extension)  # Define the path to save the image
                image_file.save(image_path)  # Save the image
                image_file_url = url_for('static', filename='img/' + filename_with_extension)  # Create the URL for the image

                try:
                    user_image = Image.open(image_file)
                    resized_image = user_image.resize((256, 256))
                    query_emb = clip_model_helper.get_single_image_embedding(resized_image, device)

                    results = chroma_db_manager.query_collection(query_emb, n_results)
                    ids_list = results['ids'][0]  # Assuming this gets the first list of ids

                    image_data = [{'url': f'/static/img/{id}.jpg', 'caption': id} for id in ids_list]
                except ValueError as e:
                    flash(str(e))

        if request.form.get('text') is not None and request.form.get('text') != '':
            query_text = request.form['text']
            query_emb = clip_model_helper.get_single_text_embedding(query_text, device)
            results = chroma_db_manager.query_collection(query_emb, n_results)
            ids_list = results['ids'][0]  # Assuming this gets the first list of ids

            # Construct URLs for each image file
            image_data = [{'url': f'/static/img/{id}.jpg', 'caption': id} for id in ids_list]

        return render_template('index.html', image_data=image_data, image_file_url=image_file_url)

    elif request.method == 'GET':
        # Simply render the template if it's a GET request
        return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4000)
