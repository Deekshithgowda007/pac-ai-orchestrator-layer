# ELK Stack uses official docker images, but we still need a Dockerfile so Compose can build
FROM docker.elastic.co/logstash/logstash:8.15.2

# Copy custom logstash pipeline configuration
COPY logstash.conf /usr/share/logstash/pipeline/logstash.conf
COPY pipelines.yml /usr/share/logstash/config/pipelines.yml

# Expose default logstash port
EXPOSE 5044
