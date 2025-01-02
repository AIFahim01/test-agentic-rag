#!/bin/bash

# Configuration
REDIS_PORT=6379
REDIS_INSIGHT_PORT=8001
CONTAINER_NAME="redis-cache"
IMAGE_NAME="redis-cache"

# Stop and remove existing container
echo "Cleaning up existing containers..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build Redis image
echo "Building Redis image..."
docker build -t $IMAGE_NAME -f redis.Dockerfile .

# Run container
echo "Starting Redis container..."
docker run -d \
   --name $CONTAINER_NAME \
   -p $REDIS_PORT:6379 \
   -p $REDIS_INSIGHT_PORT:8001 \
   -v redis_data:/data \
   $IMAGE_NAME

echo "Redis running on port $REDIS_PORT"
echo "Redis Insight available at http://localhost:$REDIS_INSIGHT_PORT"