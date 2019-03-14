import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OptionParser


object TextToParquet {

  private case class Params(
                             input: String = "",
                             fileType: String = "",
                             output: String = ""
                           )

  def read_to_df(spark: SparkSession,
                 fileType: String,
                 path: String): DataFrame = {


    val df: DataFrame = if (fileType == "txt") {
      val df_lines = spark.read.text(path)
        .withColumnRenamed("value", "content")
        .withColumn("fileName", input_file_name())
      val df_agg = df_lines
        .groupBy(col("fileName"))
        .agg(concat_ws(" ", collect_list(df_lines.col("content")))
          .as("content"))
      val df_out = df_agg.withColumn("_tmp", split(col("content"), "===="))
        .select(col("_tmp").getItem(2).as("docs"))
        .drop("_tmp")
      df_out.where(col("docs").isNotNull)
    } else {
      spark.read
        .option("header", "true")
        .option("delimiter", " ")
        .csv(path)
        .toDF("code", "docs")
    }

    println(s"${df.count()} docs processed and saved in parquet")
    df
  }

  def df_to_parquet(df: DataFrame, output_path: String): Unit = {
    df.write.mode(saveMode = "overwrite").parquet(output_path)
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName("Biomedical docs classification")
      .getOrCreate()

    Logger.getRootLogger.setLevel(Level.WARN)

    val df: DataFrame = read_to_df(spark, params.fileType, params.input)
    df_to_parquet(df, params.output)

    spark.stop()
  }


  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("ConvertFile") {
      head("Convert TXT or CSV to Parquet format")
      opt[String]("input")
        .text(s"input file. default: ${defaultParams.input}")
        .required()
        .action((x, c) => c.copy(input = x))
      opt[String]("fileType")
        .text(s"file type. default: ${defaultParams.fileType}")
        .required()
        .action((x, c) => c.copy(fileType = x))
      opt[String]("output")
        .text(s"output path. default: ${defaultParams.output}")
        .required()
        .action((x, c) => c.copy(output = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }
}



