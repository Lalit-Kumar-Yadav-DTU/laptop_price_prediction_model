from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained pipeline and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', 
                           companies=df['Company'].unique(),
                           types=df['TypeName'].unique(),
                           cpus=df['Cpu brand'].unique(),
                           gpus=df['Gpu brand'].unique(),
                           oss=df['os'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Get form data
    company = data['company']
    type_name = data['type_name']
    ram = int(data['ram'])
    weight = float(data['weight'])
    touchscreen = 1 if data['touchscreen'] == 'Yes' else 0
    ips = 1 if data['ips'] == 'Yes' else 0
    screen_size = float(data['screen_size'])
    resolution = data['resolution']
    cpu = data['cpu']
    hdd = int(data['hdd'])
    ssd = int(data['ssd'])
    gpu = data['gpu']
    os_name = data['os']

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Prepare input array (dtype=object avoids XGBoost unicode error)
    query = np.array([company, type_name, ram, weight, touchscreen,
                      ips, ppi, cpu, hdd, ssd, gpu, os_name], dtype=object)
    query = query.reshape(1, 12)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    return jsonify({'predicted_price': f'₹{predicted_price}'})

if __name__ == '__main__':
    app.run(debug=True)
