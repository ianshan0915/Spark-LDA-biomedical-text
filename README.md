# Spark-LDA-biomedical-text

LDABiotext: an LDA app for analyze biomedical literature from PMC OA Subset.

## How to use it
```
Usage: LDABiotext [options] <input>...
  --k <value>
          number of topics.
          default: 20
  --maxIterations <value>
          number of iterations of learning.
          default: 10
  --docConcentration <value>
          amount of topic smoothing to use (> 1.0) (-1=auto).  
          default: -1.0
  --topicConcentration <value>
          amount of term (word) smoothing to use (> 1.0) (-1=auto).  
          default: -1.0
  --vocabSize <value>
          number of distinct word types to use, chosen by frequency. (-1=all)  
          default: 10000
  --stopwordFile <value>
          filepath for a list of stopwords. Note: This must fit on a single machine.  
          default: null
  --algorithm <value>
          inference algorithm to use. em and online are supported.
          default: em
  --checkpointDir <value>
          Directory for checkpointing intermediate results. Checkpointing helps with recovery
          and eliminates temporary shuffle files on disk.  
          default: None
  --checkpointInterval <value>
          Iterations between each checkpoint.  Only used if checkpointDir is set.
          default: 10
   --source <value>
          Data source used, two types: text files from PMC OA subset, csv files from SparkText paper
          default: "pmc"
  <input>...
          input paths (directories) to plain text corpora.  Each text file line should hold 1 document.
          required, String
```

## Example
```
YOUR_SPARK_HOME/bin/spark-submit \
  --class "LDABiotext" \
  --master local[4] \
  target/scala-2.11/NLPIR\ 2019-assembly-1.0.jar \
  "/path/to/papers/comm_use.0-9A-B.txt/Biotechnol_B*/*.txt"
```