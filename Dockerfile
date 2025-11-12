# Recommended base image for the course
FROM jupyter/r-notebook:latest

# Ensure non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Switch to jovyan’s home (default in this image)
WORKDIR /home/jovyan/work

# Copy project files
COPY . /home/jovyan/work

# Install Python deps (none required now; keep for future)
RUN pip install --no-cache-dir -r requirements.txt || true

# Make 'make' available as entry workflow inside the container
# (Make is already present in base image; just confirm scripts are executable)
RUN chmod +x scripts/predict.py || true

# Default command: open a bash shell so users can run make targets
CMD ["/bin/bash"]
