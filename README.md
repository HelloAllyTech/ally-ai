# Lifeline AI

**Lifeline AI** is an advanced AI service designed to function as a copilot for mental health counselors. This project leverages FastAPI to deliver a robust and scalable API backend, integrating custom middleware and enhanced logging capabilities to ensure reliable performance and traceability. With environment-based configuration and containerization via Docker, Lifeline AI is built for seamless deployment and development. Managed with Poetry for dependency handling, the service provides intelligent insights and support tools to empower mental health professionals in their daily practice.

## Getting Started

### Prerequisites

- **Python 3.12** (or higher)
- **Poetry** (for dependency management)
- **Docker** (for containerization)

### Local Development


1. **Clone the Repository:**
```bash
git clone git@github.com:KeyValueSoftwareSystems/lifeline-ai.git
```

2. **Install Dependencies with Poetry:**
```bash
poetry install
```

3. **Run the Application:**

You can start the application in development mode using:

```bash
poetry run python app/main.py
```

The application will start using Uvicorn on the host and port defined in your configuration (default: http://localhost:8000).

API Documentation:

Once the application is running, you can access the automatically generated API documentation at:
    Swagger UI: http://localhost:8000/docs

### Docker

This project includes a Dockerfile to build a containerized version of the application.

Build the Docker Image:
```bash
docker build -t my-fastapi-app .
```
Run the Docker Container:
```bash
docker run -p 8000:8000 my-fastapi-app
```
The application will be available at http://localhost:8000.

