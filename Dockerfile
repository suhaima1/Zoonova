# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file to the container
COPY streamlit_app/requirements.txt .

# Install the dependencies using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code to the container
COPY . .

# Expose port 8000 for the application
EXPOSE 8501

# Set the default command to run the Python file
CMD ["streamlit", "run", "streamlit_app/zoonova_chatbot_history.py"]
