import requests

# Define the base URL where your Flask app is running
base_url = 'http://localhost:5000'  # Update with your Flask app's URL

# Verify that the Flask app is running
response = requests.get(f'{base_url}/hello')
if response.status_code == 200:
    print('Flask application is running.')
else:
    print('Flask application is not running.')

# Test the download_model endpoint
model_download_url = f'{base_url}/download_model'
response = requests.get(model_download_url)
if response.status_code == 200:
    print('Model download initiated.')
else:
    print('Model download failed.')

# Optionally, test the /api/predict endpoint (if needed)
# Replace 'BASE64_ENCODED_IMAGE_DATA' with your actual base64-encoded image data
# payload = {'image_data': 'BASE64_ENCODED_IMAGE_DATA'}
# response = requests.post(f'{base_url}/api/predict', json=payload)
# print('Prediction result:', response.json())
