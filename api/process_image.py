import os
import json
import io
from PIL import Image
from gradio_client import Client

# Initialize the Gradio client outside the handler to reuse the connection
client = Client("lllyasviel/IC-Light")

def handler(request):
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method Not Allowed'}),
            'headers': {'Content-Type': 'application/json'}
        }
    
    try:
        # Parse the multipart form data
        form = request.form
        files = request.files
        
        # Get the image file
        image_file = files.get('image')
        if not image_file:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'}),
                'headers': {'Content-Type': 'application/json'}
            }
        
        # Read the image
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Extract other parameters from the form data
        prompt = form.get('prompt', "Hello!!")
        image_width = int(form.get('image_width', 256))
        image_height = int(form.get('image_height', 256))
        images = int(form.get('images', 1))
        seed = int(form.get('seed', 3))
        steps = int(form.get('steps', 1))
        added_prompt = form.get('added_prompt', "Hello!!")
        negative_prompt = form.get('negative_prompt', "Hello!!")
        cfg_scale = float(form.get('cfg_scale', 1.0))
        highres_scale = float(form.get('highres_scale', 1.0))
        highres_denoise = float(form.get('highres_denoise', 0.1))
        lowres_denoise = float(form.get('lowres_denoise', 0.1))
        lighting_pref = form.get('lighting_pref', "None")
        
        # Prepare the input for the Gradio model
        inputs = [
            image,
            prompt,
            image_width,
            image_height,
            images,
            seed,
            steps,
            added_prompt,
            negative_prompt,
            cfg_scale,
            highres_scale,
            highres_denoise,
            lowres_denoise,
            lighting_pref
        ]
        
        # Call the Gradio model
        result = client.predict("/process_relight", inputs)
        
        # Assuming the result is a URL or base64 image, adjust as needed
        return {
            'statusCode': 200,
            'body': json.dumps({'data': result}),
            'headers': {'Content-Type': 'application/json'}
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
