package io.github.qf6101.topwords

import org.apache.spark.sql.functions.{col, sum}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by qfeng on 16-6-30.
  */

/**
  * Dictionary (initial state is overcomplete for EM)
  *
  * @param thetaS word use probability
  * @param phiS   word significance
  */
class Dictionary(val thetaS: Map[String, Double],
                 val phiS: List[(String, Double)] = Nil) extends Serializable {
  /**
    * Query certain word's theta
    *
    * @param word query word
    * @return query word's theta (returns zero if no such word found)
    */
  def getTheta(word: String): Double = {
    thetaS.getOrElse(word, 0.0)
  }

  /**
    * Test if certain word is in dictionary
    *
    * @param word query word
    * @return whether the query word in dictionary
    */
  def contains(word: String): Boolean = {
    thetaS.contains(word)
  }

  /**
    * Save the dictionary to file
    *
    * @param dictFile saving destination
    */
  def save(dictFile: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()
//    val sc = SparkContext.getOrCreate()
    // saving theta values
    spark.sparkContext.parallelize(thetaS.filter(_._1.length > 1).toList.sortBy(_._2).reverse).repartition(1).saveAsTextFile(dictFile + "/thetaS")
    // saving phi values
    if (phiS != Nil) spark.sparkContext.parallelize(phiS).repartition(1).saveAsTextFile(dictFile + "/phiS")
  }
}

object Dictionary extends Serializable {
  /**
    * Generate an overcomplete dictionary in the initial step
    * Note: using the brute force strategy which however need to use the sequential Apriori strategy instead
    *
    * @param corpus a set of texts
    * @param tauL   threshold of word length
    * @param tauF   threshold of word frequency
    * @return an overcomplete dictionary
    */
  def apply(corpus: Dataset[String],
            tauL: Int,
            tauF: Int,
            useProbThld: Double): Dictionary = {

    val sparkSession = SparkSession.builder.getOrCreate()
    import sparkSession.implicits._

    //enumerate all the possible words: corpus -> words
    val words = corpus.flatMap { text =>
      val permutations = ListBuffer[String]()
      for (i <- 1 to tauL) {
        for (j <- 0 until text.length) {
          if (j + i <= text.length) permutations += text.substring(j, j + i)
        }
      }
      permutations
    }.withColumnRenamed("value", "word")
      .groupBy("word").count().filter(row =>{
      val word = row.getString(0)
      val count = row.getLong(1)
      if(word.length == 1 || count >= tauF) true
      else false
    })
//      map(_ -> 1).reduceByKey(_ + _)
//      .filter { case (word, freq) =>
//      // leave the single characters in dictionary for smoothing reason even if they are low frequency
//      word.length == 1 || freq >= tauF
//    }
    .persist(StorageLevel.MEMORY_AND_DISK_SER_2)
    //filter words by the use probability threshold: words -> prunedWords
    val sumWordFreq = words.agg(sum("count")).first().getLong(0)
    val prunedWords = words.withColumn("prob", col("count") / sumWordFreq)
        .filter(row => {
          val word = row.getString(0)
          val prob = row.getDouble(2)
          if(word.length == 1 || prob >= useProbThld) true
          else false
        })
//      .map { case (word, freq) =>
//      (word, freq, freq / sumWordFreq)
//    }
//      .filter { case (word, _, theta) =>
//      // leave the single characters in dictionary for smoothing reason even if they have small theta
//      word.length == 1 || theta >= useProbThld
//    }
    words.unpersist()
    prunedWords.persist(StorageLevel.MEMORY_AND_DISK_SER_2)
    //normalize the word use probability: prunedWords -> normalizedWords
//    val sumPrunedWordFreq = prunedWords.map(_._2).sum()
    // 计算过滤之后的概率
    val sumPrunedWordFreq = prunedWords.agg(sum("count")).first().getLong(0)
    val data = prunedWords.withColumn("pro", col("count") / sumPrunedWordFreq)
        .select("word", "pro").collect()

    val normalizedWords = new mutable.HashMap[String, Double]
    normalizedWords.sizeHint(data.length)
    data.foreach { row => normalizedWords.put(row.getString(0), row.getDouble(1)) }

//      .map { case (word, freq, _) =>
//      word -> freq / sumPrunedWordFreq
//    }.collectAsMap().toMap
    prunedWords.unpersist()
    //return the overcomplete dictionary: normalizedWords -> dictionary
    // 获得每个单词的归一化之后的概率
    new Dictionary(normalizedWords.toMap)
  }
}
