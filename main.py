import pandas as pd


def read_data(dataset: pd.DataFrame, sep):
    df = pd.read_csv(dataset, sep=sep, header=None)
    df = pd.DataFrame(df)
    # print(len(df.columns))
    df_col = []
    for e in range(len(df.columns)):
        df_col.append('A' + str(e))
    df.columns = df_col
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df

def dataset_config(df, target_col):
    if (target_col == -1):
        feature_cols = df.columns.values.tolist()[:-1]
    else:
        feature_cols = df.columns.values.tolist()[1:]
    #print(feature_cols)
    target_col_name = df.columns.values.tolist()[target_col]
    return feature_cols, target_col_name

def conditional_prob(df:pd.DataFrame, feature_cols, target_col_name):
    df_target_grp = df.groupby(target_col_name)
    df_target = df_target_grp.size().reset_index(name='size')

    df_target['prob'] = df_target['size'] / len(df)
    df = pd.merge(df, df_target, on=target_col_name)
    for feature_col in feature_cols:
        # print(grp_name, feature_col)
        df_temp = df.groupby([feature_col, target_col_name]).size().reset_index(
            name=feature_col + "_size")
        # df_temp[feature_col+str(grp_name)+"_prob"] = df_temp[feature_col+str(grp_name)+"_size"] / df
        df = pd.merge(df, df_temp, on=[feature_col, target_col_name], how='left')


    for feature_col in feature_cols:
        df[feature_col + "_prob"] = (df[feature_col + "_size"] * df['prob']) / df['size']
    return df

def split_dataset(df, x):
    mask_test = int(df.shape[0] * (10 / 100))
    test_start = x * mask_test
    test_end = test_start + mask_test
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]], ignore_index=True)
    return train_set, test_set


def check_accuracy(test_set, train_set, feature_col, target_name):
    df_train_grp = train_set.groupby(target_name)
    suf  = df_train_grp.groups.keys()
    for grp_name, grp in df_train_grp:
        for f in feature_col:
            merge_set = grp[[f, f + "_prob"]]
            merge_set = merge_set.drop_duplicates()
            #print(merge_set)
            test_set = pd.merge(test_set, merge_set, how='left', on=[f], suffixes=suf)
            #print(train_set[[f, f + "_prob"]])
    test_set = test_set.fillna(0.01)

    list_result = []
    for grp_name, grp in df_train_grp:
        test_set['prob' + str(grp_name)] = 1
        list_result.append('prob' + str(grp_name))
        for feature in feature_col:
            test_set['prob'+str(grp_name)] *= test_set[feature+"_prob"+str(grp_name)]
    #print(list_result)

    test_set['predicted_result'] = test_set[list_result].idxmax(axis=1)
    test_set['actual_result'] = 'prob' + test_set[target_name].astype(str)
    #test_set['result'] = test_set['predicted_result'].isin(test_set['actual_result'])
    test_set['result'] = test_set.apply(lambda x: x.predicted_result in x.actual_result, axis=1)

    df_result = test_set.groupby('result')
    l = [target_name,'predicted_result','prob1', 'prob2', 'result']
    print(test_set[l])
    print("Accuracy = "+ str(df_result.size()[1]*100/(df_result.size()[1]+df_result.size()[0])) + "%")
    return test_set

if __name__ == '__main__':
    data_file = "Skin_NonSkin.txt"
    seperator = '\t'
    target_col = -1

    split_times = 1
    # num_of_grp = 10
    df: pd.DataFrame = read_data(data_file, seperator)
    #print(df)
    for x in range(split_times):
        print("Start")
        train_set, test_set = split_dataset(df, x)
        feature_cols, target_col_name  = dataset_config(train_set, target_col)
        df_all = conditional_prob(train_set,feature_cols, target_col_name)
        #print(df_all)

        test_set_op = check_accuracy(test_set, df_all, feature_cols, target_col_name)


