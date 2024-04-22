from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)


def load_data_from_csv(csv_file):
    try:
        data = pd.read_csv(csv_file)
        return data
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return None
    
# Load the trained SVM model from pickle file
with open('svm_model.pickle', 'rb') as f:
    model_SVM = pickle.load(f)

def predict_g3(school, address, higher, internet, Medu, Fedu, studytime, G1, G2):
    # Create input array for prediction
    input_data = np.array([[school, address, higher, internet, Medu, Fedu, studytime, G1, G2]])

    # Make prediction
    predicted_g3 = model_SVM.predict(input_data)
    predicted_g3 = int(predicted_g3[0])  # Ensure prediction is an integer

    # Create data for plotting
    labels = ['G1', 'G2', 'Predicted G3']
    values = [G1, G2, predicted_g3]

    # Generate and save the plot as a BytesIO object
    plt.figure(figsize=(8, 6))
    plt.plot(labels, values, label='Grades', marker='o', linestyle='-', color='blue', markersize=10)
    plt.xlabel('Grade')
    plt.ylabel('Score')
    plt.title('Comparison of G1, G2, and Predicted G3')
    plt.grid(True)
    plt.legend()

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()  # Close the plot to free up memory

    # Encode the image to base64 for embedding in HTML
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    return predicted_g3, encoded_image
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/pre')
def pre():
    return render_template('index.html')


@app.route('/compare')
def comparere():
    return render_template('compare.html')


@app.route('/dashboard')
def dash():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        school = int(request.form['school'])
        address = int(request.form['address'])
        higher = int(request.form['higher'])
        internet = int(request.form['internet'])
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        studytime = int(request.form['studytime'])
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])

        # Predict G3 and generate/save the scatter plot
        predicted_g3, encoded_image = predict_g3(school, address, higher, internet, Medu, Fedu, studytime, G1, G2)

        # Render result template with prediction and scatter plot
        return render_template('result.html', predicted_g3=predicted_g3, encoded_image=encoded_image)
    
    
# Route for generating analysis and plotting a graph
@app.route('/analysis')
def analysis():
    # Load data from the specified CSV file
    csv_file = "best 10 por - best 33 por.csv"
    data = load_data_from_csv(csv_file)

    if data is None:
        return "Error loading data. Please check the CSV file path."

    # Prepare data for plotting
    features = ['school', 'address', 'higher', 'Medu', 'studytime', 'Fedu', 'internet', 'G1', 'G2']
    target = 'G3'

    # Create individual bar plots for each feature vs. target (G3)
    num_features = len(features)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

    for i, feature in enumerate(features):
        row_index = i // 3
        col_index = i % 3
        
        # Calculate mean G3 score for each category of the feature
        feature_means = data.groupby(feature)[target].mean()

        # Plot bar graph
        feature_means.plot(kind='bar', ax=axes[row_index, col_index], alpha=0.7)
        axes[row_index, col_index].set_xlabel(feature)
        axes[row_index, col_index].set_ylabel('Mean G3')
        axes[row_index, col_index].set_title(f'{feature} vs. Mean G3')

    # Adjust layout and save the plot to a BytesIO object
    plt.tight_layout()
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Encode the plot image to base64 for embedding in HTML
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    # Render the analysis page with the encoded image
    return render_template('analysis.html', encoded_image=encoded_image)

if __name__ == '__main__':
    app.run(debug=True)
