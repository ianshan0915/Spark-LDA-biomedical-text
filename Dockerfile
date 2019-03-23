FROM alpine:3.8


ENV SPARK_VERSION=2.4.0
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.12.1 SCALA_HOME=/usr/share/scala
ENV SBT_VERSION=1.2.8
ENV SBT_HOME=/usr/local/sbt
ENV PATH ${PATH}:${SBT_HOME}/bin

RUN apk add --no-cache curl bash openjdk8-jre python3 py-pip wget git \
    #      && chmod +x *.sh \
    && mkdir /opt \
    && cd /opt \
    && wget http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install SCALA and sbt
#RUN apk add --no-cache --virtual=.build-dependencies ca-certificates \
#    && apk add --no-cache bash \
#    && cd "/tmp" \
#    && wget "https://downloads.typesafe.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz"  \
#    && tar xzf "scala-${SCALA_VERSION}.tgz"  \
#    && mkdir "${SCALA_HOME}"  \
#    && rm "/tmp/scala-${SCALA_VERSION}/bin/"*.bat \
#    && mv "/tmp/scala-${SCALA_VERSION}/bin" "/tmp/scala-${SCALA_VERSION}/lib" "${SCALA_HOME}"  \
#    && ln -s "${SCALA_HOME}/bin/"* "/usr/bin/"  \
#    && apk del .build-dependencies \
#    && rm -rf "/tmp/"* \
#    && update-ca-certificates \
RUN cd /opt \
    && wget https://sbt-downloads.cdnedge.bluemix.net/releases/v$SBT_VERSION/sbt-$SBT_VERSION.tgz \
    && mkdir -p $SBT_HOME \
    && tar xzf sbt-$SBT_VERSION.tgz \
    && ls -l \
    && mv /opt/sbt $SBT_HOME \
    && cd /opt \
    && git clone https://gitlab.com/wangxisea/spark-lda-biomedical-text.git \
    && cd spark-lda-biomedical-text \
    && sbt assembly

# Dwonload gcs connector
RUN cd /opt/spark/jars \
    && wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar

# Clean packages
RUN apk del curl git wget

WORKDIR /app
COPY target/scala-2.11/NLPIR-2019-assembly-1.3.jar /app

ENV SPARK_HOME=/opt/spark