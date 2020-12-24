from pyspark.ml.feature import Tokenizer,CountVectorizer,StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    tok = Tokenizer(inputCol = input_descript_col, outputCol = "tok")
    cv = CountVectorizer(inputCol = "tok", outputCol = output_feature_col)
    indexer = StringIndexer(inputCol = input_category_col, outputCol = output_label_col)
    pip = Pipeline(stages = [tok, cv, indexer])
    return pip

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    # record each result of different groups
    record_list = []
    
    for i in range(training_df.select("group").distinct().count()):
        test_data = training_df.filter(training_df["group"] == i)
        train_data = training_df.filter(training_df["group"] != i)
        
        pip = Pipeline(stages=[nb_0, nb_1, nb_2, svm_0, svm_1, svm_2])
        temp_model = pip.fit(train_data)
        temp_pre = temp_model.transform(test_data)
        record_list.append(temp_pre)
        
    result = record_list[0]
    # union all the result
    if len(record_list) > 1:
        for i in range(1,len(record_list)):
            result = result.union(record_list[i])
    
    train_dict = {(0.0,0.0):0.0,(0.0,1.0):1.0,(1.0,0.0):2.0,(1.0,1.0):3.0}       
    def train_value(nb,svm):
        return train_dict[(nb,svm)]
    
    train_value_udf = udf(train_value, returnType=DoubleType())
    # create 3 new column
    for i in ["0","1","2"]:
        result = result.withColumn("joint_pred_"+i, train_value_udf("nb_pred_"+i,"svm_pred_"+i))
    
            
    return result
        
        

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    input_data = base_features_pipeline_model.transform(test_df)
    input_data = gen_base_pred_pipeline_model.transform(input_data)
    
    train_dict = {(0.0,0.0):0.0,(0.0,1.0):1.0,(1.0,0.0):2.0,(1.0,1.0):3.0}
    def train_value(nb,svm):
        return train_dict[(nb,svm)]
    
    train_value_udf = udf(train_value, returnType=DoubleType())
    # create 3 new column
    for i in ["0","1","2"]:
        input_data = input_data.withColumn("joint_pred_"+i, train_value_udf("nb_pred_"+i,"svm_pred_"+i))
        
    result = gen_meta_feature_pipeline_model.transform(input_data)
    result = meta_classifier.transform(result)
    # only output the column we need
    result = result.select("id","label","final_prediction")
    
    return result
    
