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

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession, DataFrame}
import org.apache.spark.sql.functions.{input_file_name, col, concat_ws, collect_list, split, regexp_replace}

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

object LDANer {

  private case class Params(
      input: String = "",
      k: Int = 20,
      maxIterations: Int = 10,
      docConcentration: Double = -1,
      topicConcentration: Double = -1,
      vocabSize: Int = 10000,
      minDF: Int =5,
      maxDF: Double = 0.8,
      stopwordFile: String = "",
      pretrainedFolder: String = "",
      algorithm: String = "em",
      checkpointDir: Option[String] = None,
      checkpointInterval: Int = 10)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LDABiotext") {
      head("LDABiotext: an LDA app for analyze biomedical literature from PMC OA Subset.")
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
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
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
          s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[Int]("minDF")
        .text(s"the minimum number of different documents a term must appear in to be included in the vocabulary" +
          s"  default: ${defaultParams.minDF}")
        .action((x, c) => c.copy(minDF = x))
      opt[Double]("maxDF")
        .text(s"the maximum number of different documents a term must appear in to be included in the vocabulary" +
        s"  default: ${defaultParams.maxDF}")
        .action((x, c) => c.copy(maxDF = x))        
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("pretrainedFolder")
        .text(s"path for a the pretrained model NER deep learning. Note: since it is not always working online." +
        s"  default: ${defaultParams.pretrainedFolder}")
        .action((x, c) => c.copy(pretrainedFolder = x))        
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
        .text(s"Iterations between each checkpoint.  Only used if checkpointDir is set." +
        s" default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
        "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  private def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName("LDA")
      .getOrCreate()

    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
      preprocess(spark, params.input, params.vocabSize, params.minDF, params.maxDF, params.stopwordFile, params.pretrainedFolder)
    corpus.cache()
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    // Run LDA.
    val lda = new LDA()

    val optimizer = params.algorithm.toLowerCase(Locale.ROOT) match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      spark.sparkContext.setCheckpointDir(params.checkpointDir.get)
    }
    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")

    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
      }
      println()
    }
    spark.stop()
  }

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * More preprocessing is nedded.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
      spark: SparkSession,
      // paths: Seq[String],
      path: String,
      vocabSize: Int,
      minDF: Int,
      maxDF: Double,
      stopwordFile: String,
      pretrainedFolder: String): (RDD[(Long, Vector)], Array[String], Long) = {

    import spark.implicits._

    // Get corpus of document texts
  
    val df: DataFrame = spark.read
      .format("csv")
      .option("header","true")
      .option("delimiter", " ")
      .load(path)
      .toDF("code", "docs")
      .withColumn("docs", regexp_replace(col("docs"), """(['?!:\\(\\)]|\p{IsDigit}{4}|\b\p{IsLetter}{1,2}\b)\s*""", " "))

    // use spark-nlp pipeline to clean up the text
    val documentAssembler = new DocumentAssembler()
      .setInputCol("docs")
      .setOutputCol("document")
    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
    val regexTokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normalized")

    val ner = NerDLModel.load(pretrainedFolder)
      .setInputCols("normalized", "document")
      .setOutputCol("ner")
    val nerConverter = new NerConverter()
      .setInputCols("document", "normalized", "ner")
      .setOutputCol("ner_converter")
    val finisher = new Finisher()
      .setInputCols("ner_converter")
      .setCleanAnnotations(true)

    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = spark.read.textFile(stopwordFile).collect
      stopWordText.flatMap(_.stripMargin.split("\\s+"))
    }
    // val tokenizer = new RegexTokenizer()
    //   .setInputCol("docs")
    //   .setOutputCol("rawTokens")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("finished_ner_converter")
      .setOutputCol("tokens")    
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ customizedStopWords)
    val countVectorizer = new CountVectorizer()
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
      .setMaxDF(maxDF)
      .setInputCol("tokens")
      .setOutputCol("features")
    val nlp_pipeline = new Pipeline()
        .setStages(Array(
            documentAssembler,
            sentenceDetector,
            regexTokenizer,
            normalizer,
            ner,
            nerConverter,
            finisher,
            stopWordsRemover,
            countVectorizer
        ))
    val model = nlp_pipeline.fit(df)
    val documents = model.transform(df)
      .select("features")
      .rdd
      .map { case Row(features: MLVector) => Vectors.fromML(features) }
      .zipWithIndex()
      .map(_.swap)

    (documents,
      model.stages(8).asInstanceOf[CountVectorizerModel].vocabulary,  // vocabulary
      documents.map(_._2.numActives).sum().toLong) // total token count
  }
}
// scalastyle:on println
