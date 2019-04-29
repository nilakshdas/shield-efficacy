FROM rabbitmq:latest

ARG RABBITMQ_VHOSTS
ENV RABBITMQ_VHOSTS $RABBITMQ_VHOSTS

EXPOSE 5672
EXPOSE 15672

COPY enabled_plugins /etc/rabbitmq/
COPY rabbitmq.config /etc/rabbitmq/

COPY configure.sh /app/
COPY entrypoint.sh /app/
WORKDIR /app
RUN chmod +x *.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
