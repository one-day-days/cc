import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object ModelItemRecommend {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("appName").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val userZipCodePath="hdfs://hadoop102:8020/data/spark/MealRatings/userZipCode"
    val mealZipCodePath="hdfs://hadoop102:8020/data/spark/MealRatings/mealZipCode"
    //加载用户编码数据集，菜品编码数据集
    val userZipCode =sc.textFile(userZipCodePath).map{
      x=>val fields=x.slice(1,x.size-1).
        split(","); (fields(0),fields(1).toInt)}
    val mealZipCode =sc.textFile(mealZipCodePath).map{
      x=>val fields=x.slice(1,x.size-1).
        split(","); (fields(0),fields(1).toInt)}

    val reverseUserZipCode=userZipCode.map(x=>(x._2,x._1)).collect.toMap
    val reverseMealZipCode=mealZipCode.map(x=>(x._2,x._1)).collect.toMap

    import spark.implicits._

    val modelPath = "hdfs://hadoop102:8020/data/spark/MealRatings/itemModelParameters_1"
    val url ="jdbc:mysql://hadoop102:3306/test"
    val mealDF=spark.read.format("jdbc").options(
      Map( "url"-> url,
        "user" -> "root",
        "password"-> "123456",
        "dbtable" -> "meal_list")).load()
    //生成菜品数据
    val mealsMap=mealDF.select("mealID", "meal_name").
      map(row=>(row.getString(0),row.getString(1))).collect.toMap
    val trainData = sc.textFile("hdfs://hadoop102:8020/data/spark/MealRatings/trainRatings").map{
      x => val fields = x.slice(1,x.size-1).split(",");
        (fields(0).toInt,fields(1).toInt)
    }
    val trainUserRated = trainData.combineByKey(
      (x:Int)=>List(x),
      (c:List[Int],x:Int)=>x :: c,
      (c1:List [Int], c2:List[Int])=>c1:::c2).cache()
    //加载推荐模型
    val dataModel:RDD[(Int, List[(Int)])]= sc.objectFile[(Int, List[(Int)])](modelPath)
    //过滤训练数据中已有的菜品,生成可推荐的新菜品集合
    val dataModelNew = dataModel.join(trainUserRated).map(x=>(x._1,(x._2._1.diff(x._2._2))))
    //为用户(用户编码=1000)推荐10份菜品（UserNo,MealNo
    val recommendation =dataModelNew.map(x=>(x._1,x._2.take(10))).filter(x=>(x._1==1000)).flatMap(x=>x._2.map(y=>(x._1,y)))
    //反编码后的推荐结果集(UserID,MealID)
    val recommendationRecords = recommendation.map{
      case(user,meal) =>(
        reverseUserZipCode.get(user).get,
        reverseMealZipCode.get(meal).get)
    }

    //引用真实的菜品名称最终推荐结果集（UserID,MealID,Meal)
    val realRecommendationRecords =recommendationRecords.map{
      case (user, meal)=> (user,meal, mealsMap.get(meal).get)}
    realRecommendationRecords.collect.foreach(println)
  }
}
