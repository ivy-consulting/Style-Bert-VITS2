# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container 
COPY . .


# Install dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install requests_toolbelt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    python initialize.py

# Copy the FastAPI app code into the container
COPY . .

# Expose the port that your FastAPI app will run on
EXPOSE 8000

# Run the FastAPI app using the CMD command
CMD ["python", "server_fastapi.py"]
