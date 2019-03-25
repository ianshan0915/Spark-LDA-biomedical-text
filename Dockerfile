FROM alpine:3.8


ENV SPARK_VERSION=2.4.0
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.12.1
ENV SCALA_HOME=/usr/share/scala
ENV SBT_VERSION=1.2.8
ENV SBT_HOME=/usr/local/sbt
ENV PATH=${PATH}:/opt/sbt/bin
ARG CODEBASE=spark-lda-biomedical-text

WORKDIR /app

# Install Spark
RUN apk add --no-cache curl bash openjdk8-jre python3 py-pip wget git bc nss \
    && mkdir /opt \
    && cd /opt \
    && wget http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install sbt and build the jar from source code
RUN cd /opt \
    && wget https://sbt-downloads.cdnedge.bluemix.net/releases/v$SBT_VERSION/sbt-$SBT_VERSION.tgz \
    && tar xzf sbt-$SBT_VERSION.tgz \
    && sbt sbtVersion \
    && cd /opt \
    && git clone https://gitlab.com/wangxisea/${CODEBASE}.git \
    && cd ${CODEBASE} \
    && sbt assembly \
    && mv /opt/${CODEBASE}/target/scala-2.11/NLPIR-2019-assembly-1.3.jar /app \
    && rm /opt/sbt-$SBT_VERSION.tgz \
    && rm -rf /opt/sbt \
    && rm -rf /opt/${CODEBASE}

# Dwonload gcs connector
RUN cd /opt/spark/jars \
    && wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar

# Clean packages
RUN apk del curl git wget

ENV SPARK_HOME=/opt/spark