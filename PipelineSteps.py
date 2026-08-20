from sklearn.base import BaseEstimator # Remove label while fitting the data (remove outliers)

class RemoveOutliersIQR(BaseEstimator):

    def _validate_params(self):
        pass

    def __init__(self, features, threshold=1.5):
        self.features = features;
        self.threshold = threshold;

    def _fit_resample(self, X, y):
        X_clean = X.copy()
        y_clean = y.copy()

        for feature in self.features:
            Q1 = X_clean[feature].quantile(0.25)
            Q3 = X_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - self.threshold * IQR
            lim_sup = Q3 + self.threshold * IQR
            
            # Máscara booleana para manter apenas o que recai dentro dos limites
            mascara = (X_clean[feature] >= lim_inf) & (X_clean[feature] <= lim_sup)
            X_clean = X_clean[mascara]
            y_clean = y_clean[mascara]

        return X_clean, y_clean
