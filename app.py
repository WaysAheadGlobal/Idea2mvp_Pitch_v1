from flask import Flask, jsonify, render_template
import pyodbc

app = Flask(__name__)
connection_string = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=103.145.51.250;"
    "Database=NORTHWND;"
    "UID=NorthwndDB;"
    "PWD=NorthwndDB123;"
)
@app.route('/api/TelcomProducts', methods=['GET'])
def get_customers():
    try:
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute("SELECT * from TelcoDim_Products")
        columns = [column[0] for column in cursor.description]
        customers = []
        for row in cursor.fetchall():
            customer = dict(zip(columns, row))
            customers.append(customer)
        cursor.close()
        connection.close()
        return jsonify(customers)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/TelcomCategories', methods=['GET'])
def get_customers1():
    try:
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute("SELECT * from TelcoDim_Categories")
        columns = [column[0] for column in cursor.description]
        customers = []
        for row in cursor.fetchall():
            customer = dict(zip(columns, row))
            customers.append(customer)
        cursor.close()
        connection.close()
        return jsonify(customers)
    except Exception as e:
        return jsonify({'error': str(e)})
     
@app.route('/api/Products', methods=['GET'])
def get_customers2():
    try:
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute("SELECT * from tbglProducts")
        columns = [column[0] for column in cursor.description]
        customers = []
        for row in cursor.fetchall():
            customer = dict(zip(columns, row))
            customers.append(customer)
        cursor.close()
        connection.close()
        return jsonify(customers)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/Categories', methods=['GET']) 
def get_customers3():
    try:
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute("SELECT Categoryid,Categoryname,description from tbglCategories")
        columns = [column[0] for column in cursor.description]
        customers = []
        for row in cursor.fetchall():
            customer = dict(zip(columns, row))
            customers.append(customer)
        cursor.close()
        connection.close()
        return jsonify(customers)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)
