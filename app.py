from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
import os
from flask import Flask, render_template,redirect,url_for,request,session,jsonify
import pandas as pd
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np
import pyodbc
import uuid
import datetime
import smtplib
from email.mime.text import MIMEText
import random
import jwt
from datetime import timedelta
from azure.storage.blob import BlobServiceClient, BlobClient

stop_words = set(stopwords.words('english'))


# Define a list of keywords to search for
keywords = ['technology', 'innovation', 'empower', 'women', 'research', 'robotics', 'drone', 'agri', 'deep tech','mechanics', 'DPIIT', 'IIT', 'IIM', 'alumni', 'ministry', 'global', 'growth', 'seed fund', 'make in India', 'carbon', 'compliance', 'data science', 'data analytics', 'data', 'indian institute of technology', 'indian institute of management', 'skill', 'digital', 'technologies', 'million', 'billion', 'customer experience', 'organic', 'ev', 'energy', 'green energy', 'manufacturing', 'electrical vehicle']

model = api.load("word2vec-google-news-300")

key="724988cab8e24cc6ae81ebb9da8a9aa3"
# Set endpoint and key of your Form Recognizer resource
endpoint = "https://startuprecognizer.cognitiveservices.azure.com"
# Create an instance of the FormRecognizerClient
form_recognizer_client = FormRecognizerClient(endpoint, AzureKeyCredential(key))
 # Establish connection to the database
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=103.145.51.250;DATABASE=Idea2mvp_AI_ML_DB;UID=Idea2mvpdbusr;PWD=AIMLDBusr8520!')

connection_string = "DefaultEndpointsProtocol=https;AccountName=sqlvasdus3jqtllb5q;AccountKey=ppL2HciVPOQ4MGLlDfciizg7ktjfQAERDjdRLtVTOLIj5JrJYHV25M2yDqNcIiWnJVQkJtaanxHPKFiq2VkhWw==;EndpointSuffix=core.windows.net"

# Create a BlobServiceClient instance
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get the container name
container_name = "azureblob-skio-pitch-idea2mvp"


# Create a Flask app
app = Flask(__name__)
app.secret_key = 'Skio_Idea'

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')



# Secret key for encoding and decoding JWT tokens
SECRET_KEY = "06e14c7f-6278-4734-a659-f846a5cdccda"

@app.route('/generate-token', methods=['POST'])
def generate_token():
    user_id=request.form['user_id']
    license_key=request.form['license_key']
    payload = {
        'user_id': user_id,
        'license_key': license_key,
        'exp': datetime.datetime.now() + timedelta(hours=1)  # Token expiration time (adjust as needed)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return jsonify({'token': token, 'user_id': user_id})

def verify_token(user_id, license_key, token):
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        # Check if the user ID and license key in the token match the provided values
        if payload['user_id'] == user_id and payload['license_key'] == license_key:
            return True
        else:
            return False
    except jwt.ExpiredSignatureError:
        # Token has expired
        return False
    except jwt.InvalidTokenError:
        # Invalid token
        return False


@app.route('/get_otp', methods=['POST'])
def get_otp():    
    user_id = str(uuid.uuid4())
    Name = request.form['fullName']
    Email = request.form['email']
    incubator_name = request.form['incubatorName']
    Location = request.form['incubatorLocation']
    MobileNo=request.form['mobileNo']
    creation_date = datetime.datetime.now()
    cursor = conn.cursor()
    sql_query = "INSERT INTO tbgl_Customer (Name, Email, Incubator_Accelerator_Name,Location, MobileNo,CreatedDate) VALUES (?, ?, ?, ?, ?,?)"
    cursor.execute(sql_query, (Name, Email, incubator_name, Location, MobileNo,creation_date))
    conn.commit()
    otp = random.randint(100000, 999999)
    session['otp'] = otp
    session['user_id'] = user_id
    return jsonify({'otp': otp, 'user_id': user_id})


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    entered_otp = int(request.form['otp'])
    user_id = request.form['user_id']
    stored_otp = session.get('otp')
    stored_user_id = session.get('user_id')
    if entered_otp == stored_otp and user_id == stored_user_id:
        session.pop('otp', None)
        session.pop('user_id', None)
        return jsonify({'status': 'success'})
    else:
        # Return an error response
        return jsonify({'status': 'error', 'message': 'Invalid OTP or user ID'})
    
@app.route('/generate_license_key', methods=['POST'])
def generate_license_key():    
    license_key = str(uuid.uuid4())
    session['license_key'] = license_key
    
    return jsonify({'license_key': license_key})

@app.route('/request-license', methods=['POST'])
def request_license():
    Name = request.form['fullName']
    Email = request.form['email']
    incubator_name = request.form['incubatorName']
    Location = request.form['incubatorLocation']
    MobileNo=request.form['mobileNo']
    license_key = str(uuid.uuid4())  # Generate a unique license key
    is_key_used = 'No'
    creation_date = datetime.datetime.now()
    
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Define the SQL query to insert the form data into a table
    sql_query = "INSERT INTO tbgl_Customer (Name, Email, Incubator_Accelerator_Name,Location, MobileNo,LicenseKey,CreatedDate) VALUES (?, ?, ?, ?, ?,?,?)"

    # Execute the SQL query with the form data as parameters
    cursor.execute(sql_query, (Name, Email, incubator_name, Location, MobileNo,license_key,creation_date))

    # Commit the changes to the database
    conn.commit()

    # Generate OTP
    otp = random.randint(100000, 999999)

    # Send email with OTP
    sender_email = "dev@waysaheadglobal.com"
    password = "Singapore@2022"
    receiver_email = Email
    subject = "OTP Verification"
    message = f"Your OTP: {otp}"

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully")
    except smtplib.SMTPException as e:
        print("Error sending email:", str(e))

    # Store the OTP and license key in the session for validation and email sending
    session['otp'] = otp
    session['license_key'] = license_key
    session['email']= Email

    # Return a response to the user
    return redirect(url_for('verify_otp'))

# Function to store files in blob storage
def store_files_in_blob(files):
    data = []
    for uploaded_file in files:
        # Generate a unique filename to avoid overwriting
        filename = str(uuid.uuid4()) + '.pdf'

        # Create a BlobClient instance for the PDF file
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

        with uploaded_file.stream as file:
            # Upload the file to Azure Blob Storage
            blob_client.upload_blob(file, overwrite=True)
            print(f"File '{filename}' stored in the blob storage.")

            # Reset file position for further reading
            file.seek(0)

            # Read the PDF contents
            pdf_contents = file.read()

            # Analyze the PDF using the Form Recognizer API
            poller = form_recognizer_client.begin_recognize_content(pdf_contents, content_type='application/pdf')

            # Extract entities from the PDF text based on predefined keywords
            pdf_text = extract_text(file)
            entities = [keyword for keyword in keywords if keyword in pdf_text.lower()]

            # Get the result of the analysis
            result = poller.result()

            # Extract the recognized entities from the result
            for page in result:
                for line in page.lines:
                    # Split the line into words and remove stop words
                    words = line.text.split()
                    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 1 and word.isalnum()]
                    # Join the remaining words and add to entities list
                    entity = ' '.join(words)
                    entities.append(entity)

            data.append({'Filename': filename, 'Entities': str(entities)})

    return data

# Function to generate results
def generate_results(data,label):
   labels = [label.strip().lower() for label in label.split(',')]
   df=pd.DataFrame(data)
   keywords_weights = {
       'technology': 2,
       'innovation': 5,
       'empower': 3,
       'women': 4,
       'research': 2,
       'robotics': 4,
       'drone': 3,
       'agri': 2,
       'deep tech': 4,
       'DPIIT': 3,
       'IIT': 3,
       'IIM': 3,
       'alumni': 2,
       'ministry': 2,
       'global': 2,
       'growth': 3,
       'seed fund': 2,
       'make in India': 2,
       'carbon': 2,
       'compliance': 2,
       'data science': 3,
       'data analytics': 3,
       'data': 2,
       'indian institute of technology': 3,
       'indian institute of management': 3,
       'skill': 2,
       'digital': 2,
       'technologies': 2,
       'million': 2,
       'billion': 3,
       'customer experience': 2,
       'organic': 2,
       'ev': 2,
       'energy': 2,
       'green energy': 3,
       'manufacturing': 2,
       'electrical vehicle': 3
   }

   # Calculate the score for each startup
   scores = []
   for index,row in df.iterrows():
       score = 0
       for keyword, weight in keywords_weights.items():
           freq = str(row['Entities']).lower().count(keyword)
           score += weight * freq
       scores.append(score)
   
   

   # Add the scores to the dataframe and sort by score
   df['score'] = scores
   df = df.sort_values(by='score', ascending=False)
   df['rank'] = df['score'].rank(ascending=False, method='dense').astype(int)


   table_html = df[['Filename', 'score', 'rank']].to_html(index=False)

   #User filtering startups:
   filtered_startups = df[df['Entities'].apply(lambda x: any(label in x.lower() for label in labels))]
   filtered_startups = filtered_startups.assign(labels=','.join(labels))
   filtered_startups_table = filtered_startups[['Filename', 'labels']].to_html(index=False)
   
   #Clustered file
   
   labels_embeddings = []
   for labels in df['Entities']:
       labels_embedding = np.zeros(300)  # Initialize empty embedding vector
       for label in labels:
           if label in model:
               labels_embedding += model[label]
       labels_embeddings.append(labels_embedding)

   # Cluster the startups based on the similarity of their labels using K-means
   kmeans = KMeans(n_clusters=1, random_state=0).fit(labels_embeddings)
   df['Cluster'] = kmeans.labels_
   clustered_startups_table = df[['Filename', 'Cluster']].to_html(index=False)
   cluster_stats = df.groupby('Cluster').agg({'Filename': 'count', 'Entities': lambda x: list(x)})
   cluster_stats.rename(columns={'Filename': 'Count', 'Entities': 'Tags'}, inplace=True)
   
   # Add the 'Cluster' column to the cluster_stats DataFrame
   cluster_stats.reset_index(inplace=True)
   cluster_stats = cluster_stats[['Cluster', 'Count', 'Tags']]
   
   cluster_stats_table=cluster_stats[['Cluster','Count','Tags']].to_html(index=False)

  

   # Generate a unique reference number for this submission
   submission_ref = str(uuid.uuid4())

   # Add the reference number to each row in the DataFrame
   df['SubmissionRef'] = submission_ref
   filtered_startups['SubmissionRef'] = submission_ref
   cluster_stats['SubmissionRef'] = submission_ref

   # Create tables in the database to store the data
   cursor = conn.cursor()

   # Insert data into the other tables with the submission reference included
   for i, row in df.iterrows():
       cursor.execute("INSERT INTO ranked_startups (Filename, Score, Rank, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], submission_ref))
   print("Inserted in Table 1")
   for i, row in filtered_startups.iterrows():
       cursor.execute("INSERT INTO filtered_startups (Filename, Score, Rank, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], submission_ref))
   print("Inserted in Table 2")
   for i, row in df.iterrows():
       cursor.execute("INSERT INTO clustered_startups (Filename, Score, Rank, Cluster, SubmissionRef) VALUES (?, ?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], row['Cluster'], submission_ref))
   print("Inserted in Table 3")
   for i, row in cluster_stats.iterrows():
       cursor.execute("INSERT INTO cluster_stats (Cluster, Count, Startups, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Cluster'], row['Count'], ', '.join(row['Tags']), submission_ref))

   print("Inserted in all tables")
   conn.commit()

   # Close the connection
  
   conn.close()
   print("Saved in Database")

   return {
        'table_html': df[['Filename', 'score', 'rank']].to_html(index=False),
        'filtered_startups_table': filtered_startups[['Filename', 'labels']].to_html(index=False),
        'clustered_startups_table': df[['Filename', 'Cluster']].to_html(index=False),
        'cluster_stats_table': cluster_stats[['Cluster','Count','Tags']].to_html(index=False),
        'submission_ref': submission_ref
    }


# Additional API for storing files in blob
@app.route('/blob_save', methods=['POST'])
def blob_save():
    # Verify token before proceeding
    user_id = request.form['user_id']
    license_key = request.form['license_key']
    token = request.form['token']

    if verify_token(user_id, license_key, token):
        # Store files in blob
        data = store_files_in_blob(request.files.getlist('pdf_files'))

        return jsonify({'status': 'success', 'message': 'Files stored successfully', 'data': data})
    else:
        return jsonify({'status': 'error', 'message': 'Token verification failed'})

@app.route('/generate_result', methods=['POST'])
def generate_result():
    label_string = request.form['label']
    user_id = request.form['user_id']
    license_key = request.form['key']
    token = request.form['token']

    if verify_token(user_id, license_key, token):
        # Generate results
        data = store_files_in_blob(request.files.getlist('pdf_files'))
        results = generate_results(data,label_string)

        return jsonify({'status': 'success', 'message': 'Results generated successfully', 'data': results})
    else:
        return jsonify({'status': 'error', 'message': 'Token verification failed'})


@app.route('/submit', methods=['POST'])
def submit():
    label_string = request.form['label']
    labels = [label.strip().lower() for label in label_string.split(',')]
    license_key1 = request.form['key']

    data = []
    for uploaded_file in request.files.getlist('pdf_files'):
        # Generate a unique filename to avoid overwriting
        filename = str(uuid.uuid4()) + '.pdf'

        # Create a BlobClient instance for the PDF file
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

        with uploaded_file.stream as file:
            # Upload the file to Azure Blob Storage
            blob_client.upload_blob(file, overwrite=True)
            print(f"File '{filename}' stored in the blob storage.")

            # Reset file position for further reading
            file.seek(0)

            # Read the PDF contents
            pdf_contents = file.read()

            # Analyze the PDF using the Form Recognizer API
            poller = form_recognizer_client.begin_recognize_content(pdf_contents, content_type='application/pdf')

            # Extract entities from the PDF text based on predefined keywords
            pdf_text = extract_text(file)
            entities = [keyword for keyword in keywords if keyword in pdf_text.lower()]

            # Get the result of the analysis
            result = poller.result()

            # Extract the recognized entities from the result
            for page in result:
                for line in page.lines:
                    # Split the line into words and remove stop words
                    words = line.text.split()
                    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 1 and word.isalnum()]
                    # Join the remaining words and add to entities list
                    entity = ' '.join(words)
                    entities.append(entity)

            data.append({'Filename': filename, 'Entities': str(entities)})

    df=pd.DataFrame(data)
    keywords_weights = {
        'technology': 2,
        'innovation': 5,
        'empower': 3,
        'women': 4,
        'research': 2,
        'robotics': 4,
        'drone': 3,
        'agri': 2,
        'deep tech': 4,
        'DPIIT': 3,
        'IIT': 3,
        'IIM': 3,
        'alumni': 2,
        'ministry': 2,
        'global': 2,
        'growth': 3,
        'seed fund': 2,
        'make in India': 2,
        'carbon': 2,
        'compliance': 2,
        'data science': 3,
        'data analytics': 3,
        'data': 2,
        'indian institute of technology': 3,
        'indian institute of management': 3,
        'skill': 2,
        'digital': 2,
        'technologies': 2,
        'million': 2,
        'billion': 3,
        'customer experience': 2,
        'organic': 2,
        'ev': 2,
        'energy': 2,
        'green energy': 3,
        'manufacturing': 2,
        'electrical vehicle': 3
    }

    # Calculate the score for each startup
    scores = []
    for index,row in df.iterrows():
        score = 0
        for keyword, weight in keywords_weights.items():
            freq = str(row['Entities']).lower().count(keyword)
            score += weight * freq
        scores.append(score)
    
    

    # Add the scores to the dataframe and sort by score
    df['score'] = scores
    df = df.sort_values(by='score', ascending=False)
    df['rank'] = df['score'].rank(ascending=False, method='dense').astype(int)


    table_html = df[['Filename', 'score', 'rank']].to_html(index=False)

    #User filtering startups:
    filtered_startups = df[df['Entities'].apply(lambda x: any(label in x.lower() for label in labels))]
    filtered_startups = filtered_startups.assign(labels=','.join(labels))
    filtered_startups_table = filtered_startups[['Filename', 'labels']].to_html(index=False)
    
    #Clustered file
    
    labels_embeddings = []
    for labels in df['Entities']:
        labels_embedding = np.zeros(300)  # Initialize empty embedding vector
        for label in labels:
            if label in model:
                labels_embedding += model[label]
        labels_embeddings.append(labels_embedding)

    # Cluster the startups based on the similarity of their labels using K-means
    kmeans = KMeans(n_clusters=1, random_state=0).fit(labels_embeddings)
    df['Cluster'] = kmeans.labels_
    clustered_startups_table = df[['Filename', 'Cluster']].to_html(index=False)
    cluster_stats = df.groupby('Cluster').agg({'Filename': 'count', 'Entities': lambda x: list(x)})
    cluster_stats.rename(columns={'Filename': 'Count', 'Entities': 'Tags'}, inplace=True)
    
    # Add the 'Cluster' column to the cluster_stats DataFrame
    cluster_stats.reset_index(inplace=True)
    cluster_stats = cluster_stats[['Cluster', 'Count', 'Tags']]
    
    cluster_stats_table=cluster_stats[['Cluster','Count','Tags']].to_html(index=False)

   

    # Generate a unique reference number for this submission
    submission_ref = str(uuid.uuid4())

    # Add the reference number to each row in the DataFrame
    df['SubmissionRef'] = submission_ref
    filtered_startups['SubmissionRef'] = submission_ref
    cluster_stats['SubmissionRef'] = submission_ref

    # Create tables in the database to store the data
    cursor = conn.cursor()

    # Insert data into the other tables with the submission reference included
    for i, row in df.iterrows():
        cursor.execute("INSERT INTO ranked_startups (Filename, Score, Rank, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], submission_ref))
    print("Inserted in Table 1")
    for i, row in filtered_startups.iterrows():
        cursor.execute("INSERT INTO filtered_startups (Filename, Score, Rank, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], submission_ref))
    print("Inserted in Table 2")
    for i, row in df.iterrows():
        cursor.execute("INSERT INTO clustered_startups (Filename, Score, Rank, Cluster, SubmissionRef) VALUES (?, ?, ?, ?, ?)", (row['Filename'], row['score'], row['rank'], row['Cluster'], submission_ref))
    print("Inserted in Table 3")
    for i, row in cluster_stats.iterrows():
        cursor.execute("INSERT INTO cluster_stats (Cluster, Count, Startups, SubmissionRef) VALUES (?, ?, ?, ?)", (row['Cluster'], row['Count'], ', '.join(row['Tags']), submission_ref))

    print("Inserted in all tables")
    conn.commit()

    # Close the connection
   
    conn.close()
    print("Saved in Database")

    return render_template('output.html',table_html=table_html,filtered_startups_table=filtered_startups_table, clustered_startups_table= clustered_startups_table)

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
