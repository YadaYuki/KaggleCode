import pandas as pd
def generate_submission(id_name,test_id,target_name,prediction,NOTEBOOK_NAME,add_word = ""):
    from datetime import datetime
    submission_data = pd.DataFrame({id_name: test_id, target_name: prediction})
    submission_file = 'sub_'+ NOTEBOOK_NAME+ '_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '_'+add_word+'.csv'
    submission_data.to_csv(submission_file, index=False)
    print("file_name = " , submission_file)
    print(submission_data.head())
def get_mean_dict(train_df):
    fill_mean_arr = []
    null_sum = train_df.isnull().sum()
    for i in range(len(null_sum)):
        if null_sum.iloc[i] != 0:
            column = train_df.columns[i]
            mean = train_df[column].mean()
            mean_dict = {
                column : mean
            }
            fill_mean_arr.append(mean_dict)
        else:
            pass
    return fill_mean_arr

def target_encoding_regression(train_df,train_target_df,test_df,kf,encoding_columns):
    for column in encoding_columns:
        encoding_df = pd.DataFrame({
            "explain":train_df[column],
            "target":train_target_df
        })
        # encoding test_df
        # calculating mean grouped by explain variable 
        target_mean_arr = encoding_df.groupby("explain").mean()["target"]
        test_df[column] = test_df[column].map(target_mean_arr)
        # encoding train_df
        train_explain= np.repeat(np.nan,encoding_df.shape[0])
        for i1,i2 in kf.split(train_df):
            target_mean_arr_train = encoding_df.iloc[i1].groupby("explain").mean()["target"]
            train_explain[i2] = train_df.iloc[i2][column].map(target_mean_arr_train)
        train_df[column] = train_explain
    
