import dtreeviz
from sklearn.metrics import mean_squared_error
import random
from sklearn import tree
import pandas as pd
import pickle
import os
from IPython.display import display


def train_data_generate(dataset: dict, path: str):
    for key, data in dataset.items():
        data = data[data['out_y_vec'].notna()]
        data = data.rename(columns={'out_y_vec': key})
        # data = data[data['out_y_hat_vec'].notna()]
        data = data.loc[data['out_y_lb_vec'] != data['out_y_ub_vec']]
        dataset.update({key: data})

    train_data = None

    for key, data in dataset.items():
        if train_data is None:
            train_data = data[['out_date_vec', key]]
        else:
            train_data = pd.merge(train_data, data[['out_date_vec', key]], on='out_date_vec', how='left')

    train_data.dropna()
    cu_max = train_data['cu'].max()
    cu_min = train_data['cu'].min()
    al_max = train_data['al'].max()
    al_min = train_data['al'].min()

    for index, row in train_data.iterrows():
        # Assume that the lead time is given by actual operation
        cu_norm = (row['cu'] - cu_min) / (cu_max - cu_min)
        al_norm = (row['al'] - al_min) / (al_max - al_min)
        train_data.at[index, 'LT'] = random.randint(2, 3) + int(random.randint(1, 3)*cu_norm) + int(random.randint(0, 1)*al_norm)

    train_data.to_csv(path, index=False)
    print('Traning data was generated at ' + path)


def train_tree(train_path: str, save_img=False):
    data = pd.read_csv(train_path)
    # data = data.loc[:, ~data.columns.str.match("Unnamed")]
    data = data.drop(columns=['out_date_vec'])

    data.dropna(axis=0, inplace=True)

    y = data['LT']
    X = data.drop(columns=['LT'])
    column_names = X.columns.values.tolist()

    dtr = tree.DecisionTreeRegressor(max_depth=5)
    dtr.fit(X, y)
    y_pred = dtr.predict(X)
    print('MSE: {0}'.format(mean_squared_error(y, y_pred)))
    text_representation = tree.export_text(dtr, feature_names=column_names)
    print(text_representation)
    if save_img:
        viz_model = dtreeviz.model(dtr, X_train=X, y_train=y, feature_names=column_names, target_name='LT')
        v = viz_model.view()
        v.save("DT.svg")  # optionally save as svg
    return dtr


def predict_values(dataset: dict):
    for key, data in dataset.items():
        data = data[data['out_y_hat_vec'].notna()]
        data = data.loc[data['out_y_lb_vec'] != data['out_y_ub_vec']]
        data = data.drop(columns={'out_y_vec', 'out_y_lb_vec', 'out_y_ub_vec'})
        if key == 'cu':
            data = data.rename(columns={'out_y_hat_vec': key+'_mean'})
        else:
            data = data.rename(columns={'out_y_hat_vec': key})
        data = data.rename(columns={'out_y_lb_50_vec': key+'_lb'})
        data = data.rename(columns={'out_y_ub_50_vec': key + '_ub'})
        dataset.update({key: data})

    raw_data = None
    for key, data in dataset.items():
        if raw_data is None:
            if key == 'cu':
                raw_data = data[['out_date_vec', key+'_mean', key+'_lb', key+'_ub']]
            else:
                raw_data = data[['out_date_vec', key]]
        else:
            if key == 'cu':
                raw_data = pd.merge(raw_data, data[['out_date_vec', key+'_mean', key+'_lb', key+'_ub']], on='out_date_vec', how='left')
            else:
                raw_data = pd.merge(raw_data, data[['out_date_vec', key]], on='out_date_vec', how='left')

    raw_data.dropna()

    # closest_data = raw_data.iloc[raw_data.index.get_loc(datetime.strptime(target, ' %Y-%m-%d').timestamp(), method='nearest')]

    return raw_data


def make_scenarios(dtr: tree.DecisionTreeRegressor, data: pd.DataFrame, args: pd.DataFrame):
    # data = data.loc[:, ~data.columns.str.match("Unnamed")]
    # data = data.drop(columns=['out_y_vec', 'out_y_lb_vec', 'out_y_ub_vec'])

    week_cnt = 1
    snu_inputs = pd.DataFrame(columns=['PRODUCT_ID', 'SUPPLIER_ID', 'WEEK_NO', 'LEAD_TIME', 'PROBABILITY'])

    for index, row in data.iterrows():
        row_lb = row.rename({'cu_lb': 'cu'})
        row_lb = row_lb.drop(['cu_ub', 'cu_mean', 'out_date_vec'])
        df_lb = pd.DataFrame(row_lb).transpose()
        lt_lb = int(dtr.predict(df_lb))
        scenario_lb = {'PRODUCT_ID': args[args['Parameter'] == 'PRODUCT_ID'].iloc[0][1],
                       'SUPPLIER_ID': args[args['Parameter'] == 'SUPPLIER_ID'].iloc[0][1],
                       'WEEK_NO': week_cnt, 'LEAD_TIME': lt_lb,
                       'PROBABILITY': 0.25}
        snu_inputs = snu_inputs.append(scenario_lb, ignore_index=True)

        row_ub = row.rename({'cu_ub': 'cu'})
        row_ub = row_ub.drop(['cu_lb', 'cu_mean', 'out_date_vec'])
        df_ub = pd.DataFrame(row_ub).transpose()
        lt_ub = int(dtr.predict(df_ub))
        scenario_ub = {'PRODUCT_ID': args[args['Parameter'] == 'PRODUCT_ID'].iloc[0][1],
                       'SUPPLIER_ID': args[args['Parameter'] == 'SUPPLIER_ID'].iloc[0][1],
                       'WEEK_NO': week_cnt, 'LEAD_TIME': lt_ub,
                       'PROBABILITY': 0.25}
        snu_inputs = snu_inputs.append(scenario_ub, ignore_index=True)

        row_mean = row.rename({'cu_mean': 'cu'})
        row_mean = row_mean.drop(['cu_lb', 'cu_ub', 'out_date_vec'])
        df_mean = pd.DataFrame(row_mean).transpose()
        lt_mean = int(dtr.predict(df_mean))
        scenario_mean = {'PRODUCT_ID': args[args['Parameter'] == 'PRODUCT_ID'].iloc[0][1],
                         'SUPPLIER_ID': args[args['Parameter'] == 'SUPPLIER_ID'].iloc[0][1],
                         'WEEK_NO': week_cnt, 'LEAD_TIME': lt_mean,
                         'PROBABILITY': 0.5}
        snu_inputs = snu_inputs.append(scenario_mean, ignore_index=True)

        week_cnt = week_cnt + 1

    file_path = os.path.join(dir, args[args['Parameter'] == 'predict_out'].values[0][1])
    # snu_inputs = snu_inputs.loc[:, ~snu_inputs.columns.str.match("Unnamed")]
    snu_inputs.to_csv(file_path, index=False)
    print('Inputs for SNU Model were generated at ' + file_path)


if __name__ == '__main__':
    dir = os.path.dirname(__file__)
    print('Read parameters from {0}'.format(os.path.join(dir, 'args.csv')))
    args = pd.read_csv(os.path.join(dir, 'args.csv'))
    display(args)

    start_date = args[args['Parameter'] == 'START'].values[0][1]
    dataset = {}
    cu_path = os.path.join(dir, args[args['Parameter'] == 'cu_out'].values[0][1])
    dataset['cu'] = pd.read_excel(cu_path)
    ni_path = os.path.join(dir, args[args['Parameter'] == 'ni_out'].values[0][1])
    dataset['ni'] = pd.read_excel(ni_path)
    al_path = os.path.join(dir, args[args['Parameter'] == 'al_out'].values[0][1])
    dataset['al'] = pd.read_excel(al_path)
    zn_path = os.path.join(dir, args[args['Parameter'] == 'zn_out'].values[0][1])
    dataset['zn'] = pd.read_excel(zn_path)
    brent_path = os.path.join(dir, args[args['Parameter'] == 'brent_out'].values[0][1])
    dataset['brent'] = pd.read_excel(brent_path)
    dubai_path = os.path.join(dir, args[args['Parameter'] == 'dubai_out'].values[0][1])
    dataset['dubai'] = pd.read_excel(dubai_path)
    oman_path = os.path.join(dir, args[args['Parameter'] == 'oman_out'].values[0][1])
    dataset['oman'] = pd.read_excel(oman_path)
    wti_path = os.path.join(dir, args[args['Parameter'] == 'wti_out'].values[0][1])
    dataset['wti'] = pd.read_excel(wti_path)

    # train_data_generate(dataset, os.path.join(dir, args[args['Parameter'] == 'train_out'].values[0][1]))

    # dtr = train_tree(os.path.join(dir, args[args['Parameter'] == 'train_out'].values[0][1]), False)
    #
    # pickle.dump(dtr, open(os.path.join(dir, args[args['Parameter'] == 'model_out'].values[0][1]), 'wb'))

    dtr = pickle.load(open(os.path.join(dir, args[args['Parameter'] == 'model_out'].values[0][1]), 'rb'))

    future_data = predict_values(dataset)

    make_scenarios(dtr, future_data, args)
