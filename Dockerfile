# Use the official Python image
FROM python:3.12

# Set the working directory inside the server
WORKDIR /app

# Copy the requirements file and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the massive AI language model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your Python files
COPY . .

# Create the uploads folder so FastAPI doesn't panic
RUN mkdir -p uploads

# Run the FastAPI server on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
