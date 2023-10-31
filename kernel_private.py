from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class KernelTester:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        self.kernel_params = {'poly': {'degree': 3, 'coef0': 1}, 'rbf': {'gamma': 0.1}, 'sigmoid': {'coef0': 2}}
        self.results = {}
        
    def _scale_features(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def _split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42)
        
    def _test_kernels(self):
        for kernel in self.kernels:
            params = self.kernel_params.get(kernel, {})
            clf = SVC(kernel=kernel, gamma='auto', **params)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[kernel] = accuracy
            print(f'Accuracy with {kernel} kernel: {accuracy:.2f}')
    
    def _run_tests(self):
        self._scale_features()
        self._split_data()
        self._test_kernels()
        return self.results
