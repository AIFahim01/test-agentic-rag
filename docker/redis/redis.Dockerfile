FROM redis/redis-stack

WORKDIR /redis

# Redis configuration
COPY redis.conf /redis/redis.conf

# Expose ports
EXPOSE 6379
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
  CMD redis-cli ping || exit 1

# Data persistence
VOLUME ["/data"]

# Configure Redis
CMD ["redis-server", "/redis/redis.conf", "--save", "60", "1", "--loglevel", "warning"]