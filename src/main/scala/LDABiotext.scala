/*
 *
 * Author: Ian Shen
 * Date: March 9, 2019
 *
 * THis is our scala script .....

 */

// scalastyle:off println

import java.util.Locale

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{
  CountVectorizer,
  CountVectorizerModel,
  RegexTokenizer,
  StopWordsRemover
}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{
  DistributedLDAModel,
  EMLDAOptimizer,
  LDA,
  OnlineLDAOptimizer
}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

object LDABiotext {

  private case class Params(input: String = "",
                            source: String = "pmc",
                            fileType: String = "txt",
                            k: Int = 20,
                            maxIterations: Int = 10,
                            docConcentration: Double = -1,
                            topicConcentration: Double = -1,
                            vocabSize: Int = 10000,
                            minDF: Int = 5,
                            maxDF: Double = 0.8,
                            stopwordFile: String = "",
                            algorithm: String = "em",
                            checkpointDir: Option[String] = None,
                            checkpointInterval: Int = 10,
                            project_id: String = "sound-memory-230511",
                            output_bucket: String = "selm-output",
                            service_key_file: String =
                              "/opt/gitlab_hook_service_key.json")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LDABiotext") {
      head(
        "LDABiotext: an LDA app for analyze biomedical literature from PMC OA Subset.")
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(
          s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(
          s"number of distinct word types to use, chosen by frequency. (-1=all)" +
            s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[Int]("minDF")
        .text(
          s"the minimum number of different documents a term must appear in to be included in the vocabulary" +
            s"  default: ${defaultParams.minDF}")
        .action((x, c) => c.copy(minDF = x))
      opt[Double]("maxDF")
        .text(
          s"the maximum number of different documents a term must appear in to be included in the vocabulary" +
            s"  default: ${defaultParams.maxDF}")
        .action((x, c) => c.copy(maxDF = x))
      opt[String]("stopwordFile")
        .text(
          s"filepath for a list of stopwords. Note: This must fit on a single machine." +
            s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
          s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))
      opt[String]("checkpointDir")
        .text(s"Directory for checkpointing intermediate results." +
          s"  Checkpointing helps with recovery and eliminates temporary shuffle files on disk." +
          s"  default: ${defaultParams.checkpointDir}")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(
          s"Iterations between each checkpoint.  Only used if checkpointDir is set." +
            s" default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      opt[String]("source")
        .text(
          s"Data source used, two types: text files from PMC OA subset, csv files from SparkText paper" +
            s" default: ${defaultParams.source}")
        .action((x, c) => c.copy(source = x))
      opt[String]("fileType")
        .text(
          s"Data file type used, two types: text files from PMC OA subset, csv files from SparkText paper" +
            s" default: ${defaultParams.fileType}")
        .action((x, c) => c.copy(fileType = x))
      opt[String]("project_id")
        .text("GCP project ID")
        .action((x, c) => c.copy(project_id = x))
      opt[String]("output_bucket")
        .text("output bucket on GCS")
        .action((x, c) => c.copy(output_bucket = x))
      opt[String]("service_key_file")
        .text("service_key_file path")
        .action((x, c) => c.copy(service_key_file = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
          "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        // .action((x, c) => c.copy(input = c.input :+ x))
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _            => sys.exit(1)
    }
  }

  private def run(params: Params): Unit = {
    val conf = new SparkConf()
      .setAppName(s"LDABiotext with $params")
      .set("fs.gs.impl",
           "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
      .set("fs.AbstractFileSystem.gs.impl",
           "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
      preprocess(
        sc,
        params.input,
        params.source,
        params.fileType,
        params.vocabSize,
        params.minDF,
        params.maxDF,
        params.stopwordFile
      )
    corpus.cache()
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    // Run LDA.
    val lda = new LDA()

    val optimizer = params.algorithm.toLowerCase(Locale.ROOT) match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" =>
        new OnlineLDAOptimizer()
          .setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ =>
        throw new IllegalArgumentException(
          s"Only em, online are supported but got ${params.algorithm}.")
    }

    lda
      .setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }
    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble

    }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    val topics = topicIndices.map {
      case (terms, termWeights) =>
        terms.zip(termWeights).map {
          case (term, weight) => (vocabArray(term.toInt), weight)
        }
    }

    // begin of result
    save_result(
      actualCorpusSize,
      actualVocabSize,
      actualNumTokens,
      preprocessElapsed,
      elapsed,
      topics,
      params
    )

    sc.stop()
  }

  private def save_result(
      corpusSize: Float,
      vocabSize: Float,
      numTokens: Float,
      preprocessTime: Double,
      trainingTime: Double,
      topics: Array[Array[(String, Double)]],
      params: Params
  ): Unit = {

    val result_text: String =
      s"""
         |Finished training LDA model.  Summary:
         |Training time: $trainingTime sec
         |
         |Corpus summary:
         |Training time: $corpusSize sec
         |Vocabulary size: $vocabSize
         |Training set size: $numTokens tokens
         |Preprocessing time: $preprocessTime sec


         |${params.k} topics:
         |
       """.stripMargin

    val topics_result: String = ""
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        topics_result.concat(s"TOPIC $i")
        topic.foreach {
          case (term, weight) =>
            topics_result.concat(s"$term\t$weight")
        }
        topics_result.concat("\n")
    }

    println(result_text)
    println(topics_result)


    import java.time.format.DateTimeFormatter
    import java.time.LocalDateTime

    val fileName: String = DateTimeFormatter
      .ofPattern("yyyy-MM-dd_HH-mm-ss")
      .format(LocalDateTime.now)

    import scala.concurrent.ExecutionContext.Implicits.global
    import com.seancheatham.storage.gcloud.GoogleCloudStorage
    import scala.concurrent.Future

    val storage: GoogleCloudStorage = GoogleCloudStorage(
      params.project_id,
      params.service_key_file
    )

    val bytes: Iterator[Byte] = (result_text + topics_result).getBytes.toIterator
    val future: Future[_] =
      storage.write(params.output_bucket, s"sparktext-$fileName.txt")(bytes)

  }

  /**
    * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
    * More preprocessing is nedded.
    *
    * @return (corpus, vocabulary as array, total token count in corpus)
    */
  private def preprocess(
      sc: SparkContext,
      path: String,
      source: String,
      fileType: String,
      vocabSize: Int,
      minDF: Int,
      maxDF: Double,
      stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {

    val spark = SparkSession.builder
      .config("fs.AbstractFileSystem.gs.impl",
              "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
      .config("fs.gs.impl",
              "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
      .getOrCreate()
    import spark.implicits._

    // Get corpus of document texts
    // First, read one document per line in each text file, keep the filename.
    // Then aggregate the lines by filename (paper id)

    val df: DataFrame = if (source == "pmc") {
      if (fileType == "txt") {
        val df_lines =
          spark.read
            .textFile(path)
            .withColumnRenamed("value", "content")
            .withColumn("fileName", input_file_name())
        val df_agg = df_lines
          .groupBy(col("fileName"))
          .agg(
            concat_ws(" ", collect_list(df_lines.col("content"))).as("content"))
        val df_out = df_agg
          .withColumn("_tmp", split(col("content"), "===="))
          .select($"_tmp".getItem(2).as("docs"))
          .drop("_tmp")
          .withColumn(
            "docs",
            regexp_replace(
              col("docs"),
              """([?.,;!:\\(\\)]|\p{IsDigit}{4}|\b\p{IsLetter}{1,2}\b)\s*""",
              " "))
        df_out.where($"docs".isNotNull)
      } else if (fileType == "parquet") {
        spark.read.parquet(path).limit(100000)
      } else throw new IllegalArgumentException("filType was wrong...")
    } else {
      spark.read
        .format("csv")
        .option("header", "true")
        .option("delimiter", " ")
        .load(path)
        .toDF("code", "docs")
        .withColumn(
          "docs",
          regexp_replace(
            col("docs"),
            """([?.,;!:\\(\\)]|\p{IsDigit}{4}|\b\p{IsLetter}{1,2}\b)\s*""",
            " "))
    }
    println(s"±±±±±±±§§§§§§§§size is ${df.count()}, cols are ${df.columns}")

    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = sc.textFile(stopwordFile).collect()
      stopWordText.flatMap(_.stripMargin.split("\\s+"))
    }
    val tokenizer = new RegexTokenizer()
      .setInputCol("docs")
      .setOutputCol("rawTokens")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("rawTokens")
      .setOutputCol("tokens")
    stopWordsRemover.setStopWords(
      stopWordsRemover.getStopWords ++ customizedStopWords)
    val countVectorizer = new CountVectorizer()
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
      .setMaxDF(maxDF)
      .setInputCol("tokens")
      .setOutputCol("features")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model = pipeline.fit(df)
    val documents = model
      .transform(df)
      .select("features")
      .rdd
      .map { case Row(features: MLVector) => Vectors.fromML(features) }
      .zipWithIndex()
      .map(_.swap)

    (documents,
     model
       .stages(2)
       .asInstanceOf[CountVectorizerModel]
       .vocabulary,
     documents.map(_._2.numActives).sum().toLong) // total token count
  }
}

// scalastyle:on println
