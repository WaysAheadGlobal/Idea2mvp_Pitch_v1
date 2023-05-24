from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
import os
from flask import Flask, render_template,redirect,url_for,request,session
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


@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    # Retrieve the entered OTP from the form
    entered_otp = int(request.form['otp'])

    # Retrieve the stored OTP and license key from the session
    stored_otp = session.get('otp')
    license_key = session.get('license_key')

    # Check if the entered OTP matches the stored OTP
    if entered_otp == stored_otp:
        # Send email with the license key
        sender_email = "dev@waysaheadglobal.com"
        password = "Singapore@2022"
        receiver_email = session.get('email')
        subject = "License Key"
        message = f"Your license key for SKIO: {license_key}"

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

        # Clear the OTP and license key from the session
        session.pop('otp', None)
        session.pop('license_key', None)
        session.pop('email', None)

        # Return a response to the user
        return render_template('index.html')
    else:
        # Return a response to the user indicating invalid OTP
        return ('OTP Mismatch')

@app.route('/submit', methods=['POST'])
def submit():
    # Get the folder path containing the PDF files from the form data
    folder_path = request.form['folder_path']
    label_string = request.form['label'] # Get label string from input field
    labels = [label.strip().lower() for label in label_string.split(',')]
    license_key1=request.form['key']


    data=[]
    # Iterate over all PDF files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            # Get the full file path
            file_path = os.path.join(folder_path, filename)
            # Create a BlobClient instance for the PDF file
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

             
            with open(file_path, 'rb') as file:
                blob_client.upload_blob(file, overwrite=True)
                print(f"File '{filename}' stored in the blob storage.")
                
            with open(file_path, 'rb') as file:
                pdf_contents = file.read()
            

            # Analyze the PDF using the Form Recognizer API
            poller = form_recognizer_client.begin_recognize_content(pdf_contents, content_type='application/pdf')

            pdf_text = extract_text(file_path)
            entities = []
            for keyword in keywords:
                if keyword in pdf_text.lower():
                    entities.append(keyword)

            # Analyze the PDF using the Form Recognizer API
            with open(file_path, 'rb') as file:
                pdf_contents = file.read()
            poller = form_recognizer_client.begin_recognize_content(pdf_contents, content_type='application/pdf')

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
    kmeans = KMeans(n_clusters=2, random_state=0).fit(labels_embeddings)
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


    # Return the output file as a download link
    return render_template('output.html',table_html=table_html,filtered_startups_table=filtered_startups_table, clustered_startups_table= clustered_startups_table)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

