import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
# apply some styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12, 6)

def predict_and_eval(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    tr_mse = mean_squared_error(y_train, y_train_pred)
    tr_r2 = r2_score(y_train, y_train_pred)

    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Train Mean Squared Error: {tr_mse}")
    print(f"Train R-squared: {tr_r2}")

    print(f"Test Mean Squared Error: {mse}")
    print(f"Test R-squared: {r2}")

    metrics = {
        'train_mse' : tr_mse,
        'train_r2' : tr_r2,
        'test_mse' : mse,
        'test_r2' : r2
        }

    return metrics, y_train_pred, y_test_pred

class ParamTester():
    def __init__(self, model_type, X_train, y_train, X_test, y_test, train_r2_threshold=0.65):
        self.model_type = model_type
        self.best_params = {}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_r2_threshold = train_r2_threshold

    def __str__(self):
        return "ML Model Parameter Tester"
    
    def init_test_params(self, param_info : dict):
        self.param_info = param_info

    def get_attr(self, attr_name : str):
        if attr_name == 'best_params':  
            print('Current best parameters:')
            for key, value in self.best_params.items():
                print(f'{key}: {value}')
        elif attr_name == 'param_info':
            print('Parameter information:')
            for pinfo in self.param_info:
                print(f"Parameter: {pinfo['name']}, Values: {pinfo['values']}, Params: {pinfo['params']}")
        else:
            print(f'Unknown attribute: {attr_name}. Available attributes: best_params, param_info') 
        
    def test_param(self, param_name : str, params : dict, param_values : list):
        metrics = []
        for value in param_values:
            params[param_name] = value
            print(f'-- Testing {param_name} with value: {value} with {params}')
            model = self.model_type(**params)
            model.fit(self.X_train, self.y_train)
            scores, _, _ = predict_and_eval(model, 
                                            self.X_train, 
                                            self.y_train, 
                                            self.X_test, 
                                            self.y_test)
            scores[param_name] = value
            metrics.append(scores)
        return pd.DataFrame(metrics)
    
    def get_best_param(self, metrics_df, param_name : str):
        if metrics_df.empty:
            print(f'No metrics found for parameter: {param_name}. Please check the parameter values or the model.')
            return None, None
        filtered_df = metrics_df[metrics_df['train_r2'] >= self.train_r2_threshold]
        if filtered_df.empty:
            print(f'No metrics found with train R2 >= {self.train_r2_threshold} for parameter: {param_name}.')
            best_row = metrics_df.loc[metrics_df['test_r2'].idxmax()]
            best_value = best_row[param_name]
        else:
            best_row = filtered_df.loc[filtered_df['test_r2'].idxmax()]
            best_value = best_row[param_name]
        print(f'Best {param_name} value: {best_value} with test R2: {best_row["test_r2"]}')
        return best_value, best_row
    
    def update_params_with_best(self, params):
        for key, value in self.best_params.items():
            if key not in params:
                params[key] = value
        return params
    
    def plot_metrics(self, metrics_df, x_col, y_cols, title):
        for y_col in y_cols:
            plt.scatter(x=metrics_df[x_col], y=metrics_df[y_col], label=y_col)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.legend()
        plt.show()

    def run_param_tests(self):
        for pinfo in self.param_info:
            param_name = pinfo['name']
            print(f'--- Running tests for parameter: {param_name} ---')
            params = pinfo['params']
            self.update_params_with_best(params)
            print(f'Updated params: {params}')
            param_values = pinfo['values']
            metrics_df = self.test_param(param_name, params, param_values)
            best_value, best_row = self.get_best_param(metrics_df, param_name)
            # for subsequent runs, use the best_value to set the parameter
            print(f'Best {param_name} value: {best_value} : {pinfo["type"]} with test R2: {best_row["test_r2"]}')
            if pinfo['type'] == 'int':
                self.best_params[param_name] = int(best_value)
            elif pinfo['type'] == 'float':
                self.best_params[param_name] = float(best_value)
            elif pinfo['type'] == 'bool':
                self.best_params[param_name] = bool(best_value)
        
            print('-------------------------------------')    
            # Plot the metrics
            if 'train_r2' in metrics_df.columns and 'test_r2' in metrics_df.columns:
                print(f'Plotting metrics for {param_name}')
                if len(metrics_df) > 0: 
                    # Plot the metrics
                    self.plot_metrics(metrics_df, param_name, ['train_r2', 'test_r2'], f'R2 scores vs {param_name}')
            else:
                print(f'No R2 scores found for {param_name}, skipping plot.')

    