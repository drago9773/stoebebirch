# feature_selection.py

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor



class FeatureSelector:
    def __init__(self, data_path, target_column, save_path='./', estimator=None, cv=5, scoring='neg_mean_squared_error'):
        self.data_path = data_path
        self.target_column = target_column
        self.save_path = save_path
        self.estimator = estimator if estimator else ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.cv = cv
        self.scoring = scoring
        self.data = None
        self.X = None
        self.y = None
        self.selected_features = None
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        
    def preprocess_data(self):
        # Implement any necessary preprocessing here
        # For simplicity, let's assume data is ready for feature selection
        # Split data into X and y
        self.y = self.data[self.target_column]
        self.X = self.data.drop(columns=[self.target_column])
        
    def select_features(self):
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        selector = RFECV(
            estimator=self.estimator,
            step=1,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1
        )
        selector = selector.fit(self.X, self.y)
        self.selected_features = self.X.columns[selector.support_].tolist()
        print(f"Optimal number of features: {selector.n_features_}")
        print(f"Selected features: {self.selected_features}")
        
    def save_selected_features(self):
        with open(self.save_path + 'selected_features.json', 'w') as f:
            json.dump(self.selected_features, f)
        
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.select_features()
        self.save_selected_features()
        print(f"Selected features saved to {self.save_path + 'selected_features.json'}")

if __name__ == "__main__":
    selector = FeatureSelector(
        data_path='path_to_your_training_data.csv',  # Update with your actual data path
        target_column='rent',
        save_path='./'
    )
    selector.run()


# rent_predictor_base.py



class RentPredictorBase:
    def __init__(self, mode='train', save_path='./', custom_preprocess_funcs=None):
        self.mode = mode
        self.save_path = save_path
        self.selected_features = None
        self.encoder = None
        self.scaler = None
        self.poly_transformer = None
        self.knn_models = {}
        self.data = None
        self.X = None
        self.custom_preprocess_funcs = custom_preprocess_funcs or {}
        
    def load_selected_features(self):
        with open(os.path.join(self.save_path, 'selected_features.json'), 'r') as f:
            self.selected_features = json.load(f)
            
    def load_encoder_scaler_poly(self):
        self.encoder = joblib.load(os.path.join(self.save_path, 'one_hot_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(self.save_path, 'scaler.pkl'))
        self.poly_transformer = joblib.load(os.path.join(self.save_path, 'poly_transformer.pkl'))
        
    def load_knn_models(self):
        n_values = [1, 5, 10]
        for n in n_values:
            knn_filename = os.path.join(self.save_path, f'knn_model_{n}_neighbors.pkl')
            self.knn_models[n] = joblib.load(knn_filename)
        
    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        
    def preprocess_data(self):
        # Apply custom preprocessing functions if any
        for func in self.custom_preprocess_funcs.values():
            func(self.data)
        
        self._one_hot_encode_features()
        self._create_knn_benchmark_rent()
        self._fill_null()
        self._scale_features()
        self._generate_polynomial_features()
    
    def _one_hot_encode_features(self):
        columns_to_encode = ['Bedrooms', 'Bathrooms']
        if self.mode == 'train':
            self.encoder = OneHotEncoder(sparse=False, drop='first')
            encoded_features = self.encoder.fit_transform(self.data[columns_to_encode])
            joblib.dump(self.encoder, os.path.join(self.save_path, 'one_hot_encoder.pkl'))
        else:
            encoded_features = self.encoder.transform(self.data[columns_to_encode])
        self.one_hot_features = self.encoder.get_feature_names_out(columns_to_encode).tolist()
        encoded_df = pd.DataFrame(encoded_features, columns=self.one_hot_features)
        self.data.reset_index(drop=True, inplace=True)
        self.data = pd.concat([self.data, encoded_df], axis=1)
    
    def _create_knn_benchmark_rent(self):
        knn_features = ['Latitude', 'Longitude']
        n_values = [1, 5, 10]
        self.benchmark_features = []
        coords_rad = np.radians(self.data[knn_features].values)
        for n in n_values:
            column_name = f'Rent_Benchmark_{n}_neighbors'
            self.benchmark_features.append(column_name)
            if self.mode == 'train':
                knn = KNeighborsRegressor(n_neighbors=n, metric='haversine')
                knn.fit(coords_rad, self.data['rent'])
                joblib.dump(knn, os.path.join(self.save_path, f'knn_model_{n}_neighbors.pkl'))
                self.knn_models[n] = knn
            else:
                knn = self.knn_models[n]
            self.data[column_name] = knn.predict(coords_rad)
    
    def _fill_null(self):
        columns_to_fill = ['median_income']
        for column in columns_to_fill:
            fill_value = self.data[column].median()
            self.data[column] = self.data[column].fillna(fill_value)
    
    def _scale_features(self):
        potential_features = ['Square Feet', 'Bedrooms', 'Bathrooms', 'median_income'] + self.one_hot_features + self.benchmark_features
        self.data.dropna(subset=potential_features, inplace=True)
        if self.mode == 'train':
            self.scaler = MinMaxScaler()
            self.X_scaled = self.scaler.fit_transform(self.data[potential_features])
            joblib.dump(self.scaler, os.path.join(self.save_path, 'scaler.pkl'))
        else:
            self.X_scaled = self.scaler.transform(self.data[potential_features])
    
    def _generate_polynomial_features(self):
        if self.mode == 'train':
            self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
            self.X_poly = self.poly_transformer.fit_transform(self.X_scaled)
            joblib.dump(self.poly_transformer, os.path.join(self.save_path, 'poly_transformer.pkl'))
        else:
            self.X_poly = self.poly_transformer.transform(self.X_scaled)
        self.poly_feature_names = self.poly_transformer.get_feature_names_out()
        X_poly_df = pd.DataFrame(self.X_poly, columns=self.poly_feature_names)
        self.X = X_poly_df[self.selected_features]


# rent_prediction.py


class RentPredictorPredictor(RentPredictorBase):
    def __init__(self, model_path, save_path='./', custom_preprocess_funcs=None):
        super().__init__(mode='predict', save_path=save_path, custom_preprocess_funcs=custom_preprocess_funcs)
        self.model_path = model_path
        self.model = None
    
    def run(self, data_path):
        self.load_model_and_artifacts()
        self.load_data(data_path)
        self.preprocess_data()
        self.predict()
        self.save_predictions()
        print(f"Predictions saved to {os.path.join(self.save_path, 'predicted_rent.csv')}")
        
    def load_model_and_artifacts(self):
        self.model = joblib.load(os.path.join(self.model_path, 'rent_predictor_model.pkl'))
        self.load_selected_features()
        self.load_encoder_scaler_poly()
        self.load_knn_models()
        
    def predict(self):
        self.data['predicted_rent'] = self.model.predict(self.X)
    
    def save_predictions(self):
        self.data.to_csv(os.path.join(self.save_path, 'predicted_rent.csv'), index=False)

if __name__ == "__main__":
    # Define any custom preprocessing functions
    def custom_preprocess(df):
        # Add your custom preprocessing steps here
        pass

    predictor = RentPredictorPredictor(
        model_path='./',  # Update with your actual model path
        save_path='./',
        custom_preprocess_funcs={'custom_preprocess': custom_preprocess}
    )
    predictor.run(data_path='path_to_your_new_data.csv')  # Update with your actual data path


# model_training.py

class RentPredictorTrainer(RentPredictorBase):
    def __init__(self, data_path, target_column, save_path='./', custom_preprocess_funcs=None, model=None):
        super().__init__(mode='train', save_path=save_path, custom_preprocess_funcs=custom_preprocess_funcs)
        self.data_path = data_path
        self.target_column = target_column
        self.model = model if model else ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def run(self):
        self.load_data(self.data_path)
        self.load_selected_features()
        self.preprocess_data()
        self.split_data()
        self.train_model()
        self.save_model_and_artifacts()
        self.evaluate_model()
        print(f"Model saved to {os.path.join(self.save_path, 'rent_predictor_model.pkl')}")
        
    def split_data(self):
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42
        )
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def save_model_and_artifacts(self):
        joblib.dump(self.model, os.path.join(self.save_path, 'rent_predictor_model.pkl'))
    
    def evaluate_model(self):
        from sklearn.metrics import mean_squared_error
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Test MSE: {mse}")

if __name__ == "__main__":
    # Define any custom preprocessing functions
    def custom_preprocess(df):
        # Add your custom preprocessing steps here
        pass

    trainer = RentPredictorTrainer(
        data_path='path_to_your_training_data.csv',  # Update with your actual data path
        target_column='rent',
        save_path='./',
        custom_preprocess_funcs={'custom_preprocess': custom_preprocess}
    )
    trainer.run()
