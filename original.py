from flask import Flask, request, jsonify, render_template_string
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load the pre-trained BLIP model and processor for VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def answer_image_question(image, question):
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Visual Question Answering</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #f3f4f6, #e2e8f0);
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 600px;
                margin: auto;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            input[type="file"] {
                display: block;
                margin: 10px auto;
            }
            textarea {
                width: 100%;
                height: 100px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
            button {
                display: block;
                width: 100%;
                padding: 10px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            #response {
                margin-top: 20px;
                padding: 10px;
                background-color: #e7f3fe;
                border-left: 6px solid #2196F3;
                border-radius: 5px;
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visual Question Answering</h1>
            <input type="file" id="imageUpload" accept="image/*">
            <textarea id="question" placeholder="Ask a question about the image..."></textarea>
            <button id="submitButton">Submit</button>
            <div id="response"></div>
        </div>

        <script>
            document.getElementById('submitButton').addEventListener('click', async () => {
                const imageUpload = document.getElementById('imageUpload');
                const question = document.getElementById('question').value;

                if (imageUpload.files.length === 0 || question.trim() === "") {
                    alert("Please upload an image and ask a question.");
                    return;
                }

                const formData = new FormData();
                formData.append('image', imageUpload.files[0]);
                formData.append('question', question);

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    document.getElementById('response').innerText = `Answer: ${data.answer}`;
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('response').innerText = 'An error occurred. Please try again.';
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'No image or question provided'}), 400

    image_file = request.files['image']
    question = request.form['question']
    
    image = Image.open(io.BytesIO(image_file.read()))
    
    answer = answer_image_question(image, question)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
