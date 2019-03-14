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
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, StringIndexer}
import org.apache.spark.ml.classification.{NaiveBayes,LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession, DataFrame}
import org.apache.spark.sql.functions.{input_file_name, col, concat_ws, collect_list, split, regexp_replace}

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

object LabelBiodoc {

  private case class Params(
      input: String = "",
      source: String ="pmc",
      trainSize: Double = 0.7,
      maxIterations: Int = 10,
      docConcentration: Double = -1,
      topicConcentration: Double = -1,
      vocabSize: Int = 10000,
      minDF: Int =5,
      maxDF: Double = 0.8,
      stopwordFile: String = "",
      pretrainedFolder: String = "",
      algorithm: String = "nb",
      checkpointDir: Option[String] = None,
      checkpointInterval: Int = 10)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LabelBiodoc") {
      head("LabelBiodoc: a documentation classification app for analyze biomedical literature from PMC OA Subset.")
      opt[Double]("trainSize")
        .text(s"training size ratio of the total data. default: ${defaultParams.trainSize}")
        .action((x, c) => c.copy(trainSize = x))
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
      opt[String]("source")
        .text(s"Data source used, two types: text files from PMC OA subset, csv files from SparkText paper" +
        s" default: ${defaultParams.source}")
        .action((x, c) => c.copy(source = x))
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
      .appName("Biomedical docs classification")
      .getOrCreate()

    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and preprocess them for modeling.
    val preprocessStart = System.nanoTime()
    val (trainingData, testData) =
      preprocess(spark, params.trainSize, params.input, params.source, params.vocabSize, params.minDF, params.maxDF, params.stopwordFile, params.pretrainedFolder)
    trainingData.cache()
    testData.cache()
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    val trainSize = trainingData.count()
    val testSize = testData.count()

    println()
    println(s"Dataset summary:")
    println(s"\t Training set and test size: $trainSize, $testSize ")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    // Build the classification models.
    val nbmodel = new NaiveBayes()
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // choose the model based on input params, only naive bayes, logistic regression are available now
    val model_nm = params.algorithm.toLowerCase(Locale.ROOT) match {
      case "nb" => nbmodel
      case "lr" => lr
      case _ => throw new IllegalArgumentException(
        s"Only naive bayes, logistic regression are supported but got ${params.algorithm}.")
    }

    val startTime = System.nanoTime()
    val model = model_nm.fit(trainingData)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    val predictions = model.transform(testData)
    // println("The prediction of testData")
    // val preds_store = predictions.withColumn("tokens_concat", concat_ws(",", col("tokens")))
    // preds_store.select("tokens_concat","label","prediction")
    //   .write
    //   .format("csv")
    //   .save("/Users/ianshen/Documents/predictions.csv")

    // println(predictions.select("tokens_concat","label","prediction").show())
    // Evaluate the prediction results from accuracy, precision, recall
    val evaluatorAcc= new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val acc = evaluatorAcc.evaluate(predictions)
    val evaluatorPres= new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val pres = evaluatorPres.evaluate(predictions)
    val evaluatorRec= new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val rec = evaluatorRec.evaluate(predictions)

    println(s"Finished training  model.  Summary:")
    println(s"\t Training time: $elapsed sec")
    println(s"\t Test set accuracy: $acc")
    println(s"\t Test set weightedPrecision: $pres")
    println(s"\t Test set weightedRecall: $rec")

    spark.stop()
  }

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * More preprocessing is nedded.
   * @return (trainingData, testData)
   */
  private def preprocess(
      spark: SparkSession,
      train_size: Double,
      path: String,
      source: String,
      vocabSize: Int,
      minDF: Int,
      maxDF: Double,
      stopwordFile: String,
      pretrainedFolder: String): (DataFrame, DataFrame) = {

    import spark.implicits._
    // Get corpus of document texts
    // First, read one document per line in each text file, keep the filename.
    // Then aggregate the lines by filename (paper id)
  
    val df: DataFrame = if(source=="pmc") {
      val df_lines = spark.read.textFile(path).withColumnRenamed("value", "content").withColumn("fileName", input_file_name())
      val df_agg = df_lines.groupBy(col("fileName")).agg(concat_ws(" ",collect_list(df_lines.col("content"))).as("content"))
      val df_out = df_agg.withColumn("_tmp", split(col("content"), "===="))
            .select(col("_tmp").getItem(2).as("docs"))
            .drop("_tmp")
            .withColumn("docs", regexp_replace(col("docs"), """([?.,;!:\\(\\)]|\p{IsDigit}{4}|\b\p{IsLetter}{1,2}\b)\s*""", " "))
      df_out.where(col("docs").isNotNull)
    } else {
      spark.read
          .format("csv")
          .option("header","true")
          .option("delimiter", " ")
          .load(path)
          .toDF("code", "docs")
          // .withColumn("docs", regexp_replace(col("docs"), """(['?!:]|\p{IsDigit}{4}|\b\p{IsLetter}{1,2}\b)\s*""", " "))
    }

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
    // val normalizer = new Normalizer()
    //   .setInputCols("token")
    //   .setOutputCol("normalized")
    // // NerDLModel.pretrained() does not work
    // val ner = NerDLModel.load(pretrainedFolder)
    //   .setInputCols("normalized", "document")
    //   .setOutputCol("ner")
    // val nerConverter = new NerConverter()
    //   .setInputCols("document", "normalized", "ner")
    //   .setOutputCol("ner_converter")
    // val finisher = new Finisher()
    //   .setInputCols("ner_converter")
    //   .setCleanAnnotations(true)
    val stemmer = new Stemmer()
      .setInputCols("token")
      .setOutputCol("stem")
    val finisher = new Finisher()
      .setInputCols("stem")
      .setCleanAnnotations(true)
    // nlp pipeline using spark-nlp from the johnsnow labs
    val sparknlp_pipeline = new Pipeline()
        .setStages(Array(
            documentAssembler,
            sentenceDetector,
            regexTokenizer,
            stemmer,
            // normalizer,
            // ner,
            // nerConverter,
            finisher
        ))
    val df_tmp = sparknlp_pipeline.fit(Seq.empty[String].toDS.toDF("docs")).transform(df)

    // remove stop words, start to use the built-in transformers
    // add customerized stop words
    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = spark.read.textFile(stopwordFile).collect
      stopWordText.flatMap(_.stripMargin.split("\\s+"))
    }
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("finished_stem")
      .setOutputCol("tokens")    
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ customizedStopWords)
    val countVectorizer = new CountVectorizer()
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
      .setMaxDF(maxDF)
      .setInputCol("tokens")
      .setOutputCol("features")
    val indexer = new StringIndexer()
      .setInputCol("code")
      .setOutputCol("label")
    // assembly the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
          stopWordsRemover,
          countVectorizer,
          indexer
      ))

    val df_features = pipeline.fit(df_tmp).transform(df_tmp)
    val splits = df_features
      .select("label","tokens","features")
      .randomSplit(Array(train_size, 1-train_size), seed=1234)

    (splits(0), splits(1))
  }
}
// scalastyle:on println
