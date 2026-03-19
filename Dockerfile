# 1. Use a slim version of Python for a smaller, faster image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy ONLY the requirements file first (this leverages Docker caching)
COPY requirements.txt .

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. (Optional) Create a non-root user for security
RUN useradd -m myuser

# 7. Copy the rest of your application code
COPY --chown=myuser:myuser . .
RUN pip install -e .

# 8. Define the command to run your app
# Replace "main.py" with your actual entry point
USER myuser
CMD ["python", "scripts/main.py"]