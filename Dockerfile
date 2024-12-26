FROM python:3.9
WORKDIR /app

# Copy everything from the local Style-Bert-VITS2 folder to the container's Style-Bert-VITS2 folder
COPY . .

# List contents to verify
RUN ls -la

# Install requirements
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt && pip install requests_toolbelt && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && python initialize.py

# Run your FastAPI app using the CMD command
CMD ["python", "server_fastapi.py"]