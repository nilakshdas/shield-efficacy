#!/bin/bash

if [[ -z "${RABBITMQ_VHOSTS}" ]]; then
  echo "`date '+%Y-%m-%d %H:%M:%S'`.000 [error] <configure.sh> RABBITMQ_VHOSTS is not set"
  exit 1
fi

# Wait for RabbitMQ startup
for (( ; ; )); do
  sleep 1
  rabbitmqctl -q node_health_check > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "`date '+%Y-%m-%d %H:%M:%S'`.000 [info] <configure.sh> RabbitMQ is now running"
    break
  else
    echo "`date '+%Y-%m-%d %H:%M:%S'`.000 [info] <configure.sh> Waiting for RabbitMQ startup ..."
  fi
done

# Execute RabbitMQ config commands here
for vhost in $(echo ${RABBITMQ_VHOSTS} | sed "s/,/ /g"); do
  # Add vhosts
  rabbitmqctl add_vhost ${vhost}
  echo "`date '+%Y-%m-%d %H:%M:%S'`.000 [info] <configure.sh> Created vhost \"${vhost}\""

  # Set permissions for vhosts
  rabbitmqctl set_permissions -p ${vhost} admin ".*" ".*" ".*"
  echo "`date '+%Y-%m-%d %H:%M:%S'`.000 [info] <configure.sh> Permissions set for vhost \"${vhost}\""
done
