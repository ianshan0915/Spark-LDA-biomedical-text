FROM alpine:3.8


ENV SPARK_VERSION=2.4.0
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.12.1 SCALA_HOME=/usr/share/scala

RUN apk add --no-cache curl bash openjdk8-jre python3 py-pip wget git \
    #      && chmod +x *.sh \
    && mkdir /opt \
    && cd /opt \
    && wget http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install SCALA
RUN apk add --no-cache --virtual=.build-dependencies ca-certificates \
    && apk add --no-cache bash \
    && cd "/tmp" \
    && wget "https://downloads.typesafe.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz"  \
    && tar xzf "scala-${SCALA_VERSION}.tgz"  \
    && mkdir "${SCALA_HOME}"  \
    && rm "/tmp/scala-${SCALA_VERSION}/bin/"*.bat \
    && mv "/tmp/scala-${SCALA_VERSION}/bin" "/tmp/scala-${SCALA_VERSION}/lib" "${SCALA_HOME}"  \
    && ln -s "${SCALA_HOME}/bin/"* "/usr/bin/"  \
    && apk del .build-dependencies \
    && rm -rf "/tmp/"* \
    && update-ca-certificates \
    && curl -fsL https://github.com/sbt/sbt/releases/download/v$SBT_VERSION/sbt-$SBT_VERSION.tgz | tar xfz - -C /usr/local \
    && $(mv /usr/local/sbt-launcher-packaging-$SBT_VERSION /usr/local/sbt || true) \
    && ln -s /usr/local/sbt/bin/* /usr/local/bin/  \
    && apk del curl git wget

# Dwonload gcs connector
RUN cd /opt/spark/jars \
    && wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar

# Build jar
RUN cd /opt \
    && git clone https://gitlab.com/wangxisea/spark-lda-biomedical-text.git \
    && cd spark-lda-biomedical-text \
    && sbt assembly

WORKDIR /app
COPY target/scala-2.11/NLPIR-2019-assembly-1.3.jar /app

ENV SPARK_HOME=/opt/spark