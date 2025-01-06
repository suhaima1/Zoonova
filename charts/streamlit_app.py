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
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key
client = OpenAI()

# Function to encode the image
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')
# def encode_image(image):
#     with io.BytesIO() as buffer:
#         image.save(buffer, format="PNG")
#         return base64.b64encode(buffer.getvalue()).decode('utf-8')
# Path to your image
# image_path = "./Image_1.png"

# Getting the base64 string
# base64_image = encode_image(image_path)
# Function to process text data

def process_text_data(text_data):
    return text_data.strip()

def generate_charts(input):
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
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": f"""Data: {input}
    #             Analyze the data that is provided in the input and recommend suitable charts.
    #             Return your response with python code that uses matplotlib to generate those charts.
    #             Please note that the code should be complete with data so that when i run the code on my system, 
    #             it saves images."""
    #         }
    #     ],
    #     max_tokens=2000,
    # )

    # # Example response message content
    # response_message = response.choices[0].message.content

    # Use regex to find code blocks
    # code_blocks = re.findall(r"```python(.*?)```", response_message, re.DOTALL)

    # Save the extracted code to a file or process it
    for i, code in enumerate(code_blocks):
        function_code = f"""
def generated_chart_function():
{code.strip().replace('\n', '\n    ')}
"""
        try:
            # Execute the function code
            exec(function_code, globals())
            # Call the generated function
            generated_chart_function()

            st.write("Code executed successfully.")

            # Find generated image files
            image_files = [f for f in os.listdir('./') if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                st.write(f"Found {len(image_files)} image(s) in the directory.")
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
        # file_name = f"extracted_code_{i + 1}.py"
        # with open(file_name, "w") as code_file:
        #     code_file.write(code.strip())
        #     st.write('Code written to file')
        # print(f"Code saved to {file_name}")

    # Run the generated Python file
    # try:
    #     print(f"Running {file_name}...")
    #     result = subprocess.run(
    #         ["python", file_name], capture_output=True, text=True
    #     )
    #     print(f"Output from {file_name}:\n{result.stdout}")
    #     st.write(f"Output from {file_name}:\n{result.stdout}")
    #     if result.stderr:
    #         print(f"Error from {file_name}:\n{result.stderr}")
    #         st.write(f"Error from {file_name}:\n{result.stderr}")
    # except Exception as e:
    #     print(f"An error occurred while running {file_name}: {e}")
    #     st.write(f"An error occurred while running {file_name}: {e}")


def main():
    st.title("Zoonova Charts")
    import matplotlib
    st.write(f"Matplotlib is installed: {matplotlib.__version__}")
    # uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    input = st.text_input("Enter:", "")

# Submit button
    if st.button("Submit"):
        # Display the uploaded image
        # image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)

        # Encode the image
        # base64_image = encode_image(image)
        
        # Generate and display charts
        st.write("### Generated Charts")

        if input is not None:
            generate_charts(input)
            image_files = [f for f in os.listdir('./') if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                print(f"Found {len(image_files)} image(s) in the directory.")
                st.write(f"Found {len(image_files)} image(s) in the directory.")
                # Display all images
                for img_file in image_files:
                    image_path = os.path.join('./', img_file)
                    image = Image.open(image_path)
                    st.image(image, caption=img_file, use_column_width=True)
                    os.remove(image_path)
            else:
                st.write("No charts generated.")
        else:
            st.write('No data entered.')

if __name__ == "__main__":
    main()
