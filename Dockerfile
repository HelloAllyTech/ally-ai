# Use the official Python 3.12 slim image as the base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and tools needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry using its official installer
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition files to leverage Docker layer caching
COPY pyproject.toml poetry.lock* /app/

# Install only the production dependencies (adjust flags as needed)
RUN poetry install --no-root --only main

# Copy the entire application code to the container
COPY . /app

# Expose the port on which the app will run
EXPOSE 8000

# Set the default command to run your FastAPI app.
# This command will execute the "if __name__ == '__main__'" block in app/main.py,
# which calls uvicorn.run() with the specified settings.
CMD ["poetry", "run", "python", "app/main.py"]
