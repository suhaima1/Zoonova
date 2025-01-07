from openai import OpenAI
# from dotenv import load_dotenv
import os
import base64
import re
import subprocess
import streamlit as st
from PIL import Image
import io
import openai
import matplotlib.pyplot as plt
# Load environment variables from .env file
# load_dotenv()

# Access the API key
# api_key = os.getenv("OPENAI_API_KEY")
api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def encode_image(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Function to process text data
def process_text_data(text_data):
    return text_data.strip()
    
def generate_charts(input, type):
    if type == 'text':
        processed_text = process_text_data(input)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the following text data and recommend suitable charts.
                    Return your response with python code that uses matplotlib and mplfinance libraries to generate those charts.
                    Please note that the code should be complete with data so that when i run the code on my system, it saves images. 
                    The text data is: {processed_text}""",
                }
            ],
            max_tokens=4000,
        )
        response_message = response.choices[0].message.content
    
        # Use regex to find code blocks
        code_blocks = re.findall(r"```python(.*?)```", response_message, re.DOTALL)
    
    elif type == 'image':
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": """Analyze the data that is provided in the image and recommend suitable charts.
                Return your response with python code that uses matplotlib and mplfinance to generate those charts.
                Please note that the code should be complete with data so that when i run the code on my system, 
                it saves images."""},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{input}",
                },
                },
            ],
            }
        ],
        max_tokens=2000,
        )

        # Example response message content
        response_message = response.choices[0].message.content

        # Use regex to find code blocks
        code_blocks = re.findall(r"```python(.*?)```", response_message, re.DOTALL)

    # Save the extracted code to a file or process it
    for i, code in enumerate(code_blocks):
        function_code = """
def generated_chart_function():
    {}
""".format(code.strip().replace('\n', '\n    '))
        try:
            # Execute the function code
            exec(function_code, globals())
            
            # Call the generated function
            generated_chart_function()

            # Find generated image files
            image_files = [f for f in os.listdir('./') if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                # Display all images
                for img_file in image_files:
                    image_path = os.path.join('./', img_file)
                    image = Image.open(image_path)
                    st.image(image, caption=img_file, use_column_width=True)
                    os.remove(image_path)  # Clean up after displaying
            else:
                st.write("No charts generated.")
        except Exception as e:
            st.write(f"An error occurred: {e}")

def main():
    st.title("Zoonova Charts")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    input = st.text_area(label="", placeholder="Enter text:", height=30)

    if st.button("Submit"):
         # Generate and display charts
        st.write("### Generated Charts")
        # Display the uploaded image
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Encode the image
            base64_image = encode_image(image)
            generate_charts(base64_image, 'image')

        elif input:
            generate_charts(input, 'text')

        else:
            st.write('No data entered.')

if __name__ == "__main__":
    main()
