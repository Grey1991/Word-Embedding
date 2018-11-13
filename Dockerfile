# Basic
FROM ubuntu
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 LAN=en

# Dependencies
RUN apt-get update && apt-get install -y wget bzip2 unzip libgomp1

# Setup Anaconda 3
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

# Update PATH
ENV PATH /opt/conda/bin:$PATH

# Install add-apt-repository
RUN apt-get install -y software-properties-common python-software-properties

# Install Java.
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    add-apt-repository -y ppa:webupd8team/java && \
    apt-get update && \
    apt-get install -y oracle-java8-installer && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer

# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

RUN conda install numpy==1.12.1
RUN conda install pandas==0.21.0
RUN conda install tensorflow==1.0.1
RUN conda install gensim==3.0.1

# Install Spacy
RUN conda install spacy==1.8.2
RUN python3 -m spacy.${LAN}.download all

# Copy all stuff to root.
ADD . /root

# Set working directory
WORKDIR /root

# Run the test
CMD [ "/bin/bash", "-c", "python testcode.py 100001"]

