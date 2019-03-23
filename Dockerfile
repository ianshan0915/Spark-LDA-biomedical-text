FROM alpine:3.8


ENV SPARK_VERSION=2.4.0
ENV HADOOP_VERSION=2.7

RUN apk add --no-cache curl bash openjdk8-jre python3 py-pip wget \
    #      && chmod +x *.sh \
    && mkdir /opt \
    && cd /opt \
    && wget http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Dwonload gcs connector
RUN cd /opt/spark/jars \
    && wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar


WORKDIR /app
RUN sbt assembly
COPY target/scala-2.11/NLPIR-2019-assembly-1.3.jar /app

ENV SPARK_HOME=/opt/spark