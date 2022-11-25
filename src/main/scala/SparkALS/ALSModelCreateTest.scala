package SparkALS

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

object ALSModelCreateTest {
  val appName = "Create ALS ModelTest "
  val conf = new SparkConf().setAppName(appName)
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")

  def main(args: Array[String]) = {
    if (args.length != 6) {
      System.err.println("Usage: ALSModelCreate requires: 6 input fields <trainDataPath> <modelPath>  " +
        "<rank> <iteration> <lambda> <splitter>")
    }


    // 匹配输入参数
    val trainDataPath = args(0)
    val modelPath = args(1)
    val rank = args(2).toInt
    val iteration = args(3).toInt
    val lambda = args(4).toDouble
    val splitter = args(5)

    // 加载训练集数据
    val trainData = sc.textFile(trainDataPath).map{x=>val fields=x.slice(1,x.size-1).split(splitter);
      (fields(0).toInt,fields(1).toInt,fields(2).toDouble)}
    val trainDataRating= trainData.map(x=>Rating(x._1,x._2,x._3))

    // 建立ALS模型
    val model = ALS.train(trainDataRating, rank, iteration, lambda)
    // 存储ALS模型
    model.save(sc,modelPath)
    println("Model saved")
    sc.stop()
  }

}
