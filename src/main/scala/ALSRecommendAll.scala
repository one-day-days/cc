import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object ALSRecommendAll {

  def main(args: Array[String]): Unit = {
    //设置输入参数
    val conf = new SparkConf().setAppName("appName").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    import spark.implicits._

    val userZipCodePath="hdfs://master:8020/data/spark/MealRatings/userZipCode"
    val mealZipCodePath="hdfs://master:8020/data/spark/MealRatings/mealZipCode"
    val trainDataPath="hdfs://master:8020/data/spark/MealRatings/trainRating"
    val modelPath = "hdfs://master:8020/data/spark/MealRatings/MedolALS"
    val recommendationsPath="hdfs://master:8020/data/spark/MealRatings/ALSRecommendations"
    val splitter=","
    val K=10
    //从外部数据库(MySQL)，加载菜品数据
    val url = "jdbc:mysql://hadoop102:3306/test"
    val mealDF = spark.read.format("jdbc").options(
      Map("url" -> url,
        "user" -> "root" ,
        "password" -> "123456",
        "dbtable" -> "meal_list"

      )).load()

    //生成菜品数据
    val mealsMap = mealDF.select( "mealID", "meal_name").
      map(row=> (row.getString(0), row.getString(1))).collect.toMap

    //加载用户编码数据集，菜品编码数据集
    val userZipCode=sc.textFile(userZipCodePath).map{
      x=>val fields=x.slice(1,x.size-1).
        split(splitter); (fields(0),fields(1).toInt)
    }


    val mealZipCode=sc.textFile(mealZipCodePath).map{
      x=>val fields=x.slice(1,x.size-1).
        split(splitter); (fields(0),fields(1).toInt)
    }

    //对推荐结果集中的用户与菜品编码进行反规约操作
    //(UserNo->UserID, MealNO->MealID)
    val reverseUserZipCode=userZipCode.map(x=>(x._2,x._1)).collect.toMap
    val reverseMealZipCode=mealZipCode.map(x=>(x._2,x._1)).collect.toMap
    //加载 ALS model
    val model=MatrixFactorizationModel.load(sc, modelPath)
    model.productFeatures.cache
    model.userFeatures.cache
    println("model retrieved.")
    //加载训练数据
    val trainData= sc.textFile(trainDataPath).map{
      x=>val fields=x.slice(1,x.size-1).
        split(splitter);((fields(0).toInt,fields(1).toInt),fields(2).toDouble)
    }
    //向所有用户推荐100份菜品
    val recommendationList=model.recommendProductsForUsers(100).map(x=>x._2)
    val recommendRecords=recommendationList.flatMap(x=>{
      for(i<-0 until x.length-1) yield (x(i))}).map{
      case Rating(user,product,rating)=>((user,product),rating)
    }
    //过滤训练数据中已有的菜品,生成可推荐的新菜品集合
    val recomendNewRecordsAll=recommendRecords.subtractByKey(trainData)
    //为所有用户推荐菜品,按预测评分倒序取K份菜品(UserNo, MealNo)
    val recomendNewRecordsK=recomendNewRecordsAll.map{
      case ((user,meal),rating)=>(user,meal,rating)
    }.sortBy(x => (x._1,x._3),false).
      map(x=>(x._1,x._2)).
      combineByKey(
        (x:Int) =>List(x),
        (c:List[Int],x:Int) => c :+ x,
        (c1:List[Int], c2:List[Int])=>c1 ::: c2).
      map(x => (x._1,x._2.take(K))).flatMap(x=>x._2.map(y=>(x._1,y))
    )
    //反编码后的推荐结果集（UserID,MealID)
    val recommendationRecords=recomendNewRecordsK.map{
      case(user,meal) =>(reverseUserZipCode.get(user).get,
        reverseMealZipCode.get (meal).get)
    }
    //引用真实的菜品名称,最终推荐结果集（UserID,MealID, Meal)
    val realRecommendationRecords =recommendationRecords.map{
      case (user,meal) => (user,meal,mealsMap.get(meal).get)
    }

    realRecommendationRecords.collect.foreach(println)

    //存储推荐结果集
    realRecommendationRecords.saveAsTextFile(recommendationsPath)
  }

}
