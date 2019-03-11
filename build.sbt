name := "NLPIR 2019"

version := "1.2"

scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0" % "provided"
libraryDependencies += "com.github.scopt" %% "scopt"% "4.0.0-RC2"
libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.8.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0" % "provided"

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}