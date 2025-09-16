import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox, 
    QPushButton, QHBoxLayout, QStackedWidget, QRadioButton, QLineEdit, 
    QComboBox, QFormLayout, QFileDialog, QTabWidget, QGroupBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QProgressBar,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import shap
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== الدوال الجاهزة التي سيتم استدعاؤها ====================

def predict_function(model, data):
    """دالة التنبؤ الجاهزة"""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities

def explain_function(model, data):
    """دالة التفسير الجاهزة"""
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values

# ==================== نهاية الدوال الجاهزة ====================

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object, object, object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, target_column, test_size=0.2):
        super().__init__()
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size

    def run(self):
        try:
            # Load data
            self.progress_signal.emit(10)
            df = pd.read_csv(self.file_path)
            
            # Check if target column exists
            if self.target_column not in df.columns:
                self.error_signal.emit(f"Target column '{self.target_column}' not found in data")
                return
            
            # Preprocessing
            self.progress_signal.emit(30)
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Convert categorical variables if any
            label_encoders = {}
            for column in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le
            
            # Split data
            self.progress_signal.emit(50)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )
            
            # Train model
            self.progress_signal.emit(70)
            model = XGBClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            
            # استدعاء دالة التنبؤ الجاهزة
            self.progress_signal.emit(85)
            y_pred, _ = predict_function(model, X_test)
            
            # Calculate metrics
            self.progress_signal.emit(95)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            self.progress_signal.emit(100)
            self.finished_signal.emit(model, X_test, y_test, accuracy, report)
            
        except Exception as e:
            self.error_signal.emit(str(e))

class PredictionThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def run(self):
        try:
            self.progress_signal.emit(50)
            # استدعاء دالة التنبؤ الجاهزة
            predictions, probabilities = predict_function(self.model, self.data)
            
            self.progress_signal.emit(100)
            self.finished_signal.emit(predictions, probabilities)
            
        except Exception as e:
            self.error_signal.emit(str(e))

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Failure Prediction System")
        self.resize(1200, 800)
        self.setStyleSheet(self.style())
        
        self.model = None
        self.X_test = None
        self.y_test = None
        self.accuracy = None
        self.current_data = None
        
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.root = QVBoxLayout(central)
        self.stack = QStackedWidget()
        self.root.addWidget(self.stack, stretch=1)

        self.btn_layout = QHBoxLayout()
        self.next = QPushButton("Next")
        self.back = QPushButton("Back")
        self.btn_layout.addWidget(self.back)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.next)
        self.root.addLayout(self.btn_layout)

        self.pages = []
        self.pages.append(self.welcome_page())
        self.pages.append(self.train_data_page())
        self.pages.append(self.predict_page())
        self.pages.append(self.report_page())

        for page in self.pages:
            self.stack.addWidget(page)
        self.stack.setCurrentIndex(0)

        self.next.clicked.connect(self.go_next)
        self.back.clicked.connect(self.go_back)
        self.update_buttons()

    def go_back(self):
        current_index = self.stack.currentIndex()
        if current_index > 0:
            self.stack.setCurrentIndex(current_index - 1)
        self.update_buttons()
    
    def go_next(self):
        current_index = self.stack.currentIndex()
        if current_index < len(self.pages) - 1:
            self.stack.setCurrentIndex(current_index + 1)
        self.update_buttons()

    def update_buttons(self):
        current_index = self.stack.currentIndex()
        self.back.setEnabled(current_index > 0)
        if current_index == len(self.pages) - 1:
            self.next.setText('Finish')
            self.next.clicked.disconnect()
            self.next.clicked.connect(self.close)
        else:
            self.next.setText('Next')
            self.next.setEnabled(True)
        if current_index == 0:
            self.back.setEnabled(False)

    def welcome_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Machine Failure Prediction System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2E86AB; margin: 20px;")
        layout.addWidget(title)

        desc = QLabel(
            "This system helps you predict machine failures using machine learning. "
            "You can either train a new model using your data or use a pre-trained model for predictions."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(desc)
        
        # Options group
        options_group = QGroupBox("Select an option to continue")
        options_layout = QVBoxLayout(options_group)
        
        self.radio1 = QRadioButton("Train a new model")
        self.radio2 = QRadioButton("Use a pre-trained model")
        
        options_layout.addWidget(self.radio1)
        options_layout.addWidget(self.radio2)
        
        # New model options
        self.newmodel_layout = QVBoxLayout()
        self.newmodel_container = QWidget()
        self.newmodel_container.setLayout(self.newmodel_layout)
        self.newmodel_container.setVisible(False)
        options_layout.addWidget(self.newmodel_container)
        
        layout.addWidget(options_group)
        layout.addStretch()
        
        # Connect signals
        self.radio1.toggled.connect(self.radio_select)
        self.radio2.toggled.connect(self.radio_select)

        return page

    def radio_select(self):
        if self.radio1.isChecked():
            self.newmodel_container.setVisible(True)
            self.new_model_options()
        else:
            self.newmodel_container.setVisible(False)
            self.pretrained_model_options()

    def clear_layout(self, layout):
        """Remove all widgets from the layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            sub_layout = item.layout()
            if sub_layout is not None:
                self.clear_layout(sub_layout)

    def new_model_options(self):
        self.clear_layout(self.newmodel_layout)
        form_layout = QFormLayout()
        
        # Data file selection
        file_layout = QHBoxLayout()
        self.data_file = QLineEdit()
        self.data_file.setPlaceholderText("Select data CSV file...")
        self.data_browse = QPushButton("Browse")
        self.data_browse.clicked.connect(self.browse_data_file)
        file_layout.addWidget(self.data_file)
        file_layout.addWidget(self.data_browse)
        form_layout.addRow("Data File:", file_layout)
        
        # Target column selection
        self.target_column = QComboBox()
        form_layout.addRow("Target Column:", self.target_column)
        
        # Test size
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.2)
        form_layout.addRow("Test Size:", self.test_size)
        
        self.newmodel_layout.addLayout(form_layout)

    def pretrained_model_options(self):
        self.clear_layout(self.newmodel_layout)
        form_layout = QFormLayout()
        
        # Model file selection
        file_layout = QHBoxLayout()
        self.model_file = QLineEdit()
        self.model_file.setPlaceholderText("Select trained model file...")
        self.model_browse = QPushButton("Browse")
        self.model_browse.clicked.connect(self.browse_model_file)
        file_layout.addWidget(self.model_file)
        file_layout.addWidget(self.model_browse)
        form_layout.addRow("Model File:", file_layout)
        
        self.newmodel_layout.addLayout(form_layout)
        self.go_next()

    def browse_data_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            self.data_file.setText(file_name)
            self.load_data_columns(file_name)

    def browse_model_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.json *.model);;All Files (*)", options=options
        )
        if file_name:
            self.model_file.setText(file_name)
            self.load_model(file_name)

    def load_data_columns(self, file_path):
        try:
            df = pd.read_csv(file_path, nrows=1)
            self.target_column.clear()
            self.target_column.addItems(df.columns.tolist())
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")

    def load_model(self, file_path):
        try:
            if file_path.endswith('.json'):
                self.model = XGBClassifier()
                self.model.load_model(file_path)
                QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model: {str(e)}")

    def train_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Model Training")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Train button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setVisible(False)
        layout.addWidget(self.results_text)
        
        layout.addStretch()
        return page

    def start_training(self):
        if not self.data_file.text():
            QMessageBox.warning(self, "Error", "Please select a data file first!")
            return
        
        if not self.target_column.currentText():
            QMessageBox.warning(self, "Error", "Please select a target column!")
            return
        
        self.progress_bar.setVisible(True)
        self.train_btn.setEnabled(False)
        
        self.thread = TrainingThread(
            self.data_file.text(),
            self.target_column.currentText(),
            self.test_size.value()
        )
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.error_signal.connect(self.training_error)
        self.thread.start()

    def training_finished(self, model, X_test, y_test, accuracy, report):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = accuracy
        
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        self.results_text.setVisible(True)
        
        results = f"Training Completed!\n\n"
        results += f"Accuracy: {accuracy:.4f}\n\n"
        results += "Classification Report:\n"
        results += report
        results += "\n\nModel is ready for predictions!"
        
        self.results_text.setText(results)
        QMessageBox.information(self, "Success", "Model trained successfully!")

    def training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Training Error", error_msg)

    def predict_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Make Predictions")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Data input section
        input_group = QGroupBox("Input Data for Prediction")
        input_layout = QVBoxLayout(input_group)
        
        self.data_input = QTextEdit()
        self.data_input.setPlaceholderText("Enter data in CSV format or load from file...")
        input_layout.addWidget(self.data_input)
        
        # Load data button
        load_btn = QPushButton("Load Data from File")
        load_btn.clicked.connect(self.load_prediction_data)
        input_layout.addWidget(load_btn)
        
        layout.addWidget(input_group)
        
        # Predict button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_btn)
        
        # Results area
        self.prediction_results = QTextEdit()
        self.prediction_results.setVisible(False)
        layout.addWidget(self.prediction_results)
        
        layout.addStretch()
        return page

    def load_prediction_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Prediction Data", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            try:
                df = pd.read_csv(file_name)
                self.data_input.setText(df.to_csv(index=False))
                self.current_data = df
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")

    def make_prediction(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model available! Please train or load a model first.")
            return
        
        data_text = self.data_input.toPlainText().strip()
        if not data_text:
            QMessageBox.warning(self, "Error", "Please enter some data to predict!")
            return
        
        try:
            # Parse input data
            from io import StringIO
            df = pd.read_csv(StringIO(data_text))
            self.current_data = df
            
            # استدعاء دالة التنبؤ الجاهزة فقط
            predictions, probabilities = predict_function(self.model, df)
            
            # Display results
            results = "Prediction Results:\n\n"
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results += f"Sample {i+1}: Prediction = {pred}, Probability = {max(prob):.4f}\n"
            
            self.prediction_results.setText(results)
            self.prediction_results.setVisible(True)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Prediction failed: {str(e)}")

    def report_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Model Report and Explanation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Tabs for different reports
        tabs = QTabWidget()
        
        # Model performance tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)
        self.performance_text = QTextEdit()
        perf_layout.addWidget(self.performance_text)
        tabs.addTab(perf_tab, "Performance")
        
        # SHAP explanation tab
        shap_tab = QWidget()
        shap_layout = QVBoxLayout(shap_tab)
        self.shap_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        shap_layout.addWidget(self.shap_canvas)
        tabs.addTab(shap_tab, "SHAP Explanation")
        
        layout.addWidget(tabs)
        
        # Generate report button
        self.report_btn = QPushButton("Generate Report")
        self.report_btn.clicked.connect(self.generate_report)
        layout.addWidget(self.report_btn)
        
        return page

    def generate_report(self):
        if self.model is None or self.X_test is None:
            QMessageBox.warning(self, "Error", "No model or test data available!")
            return
        
        # Generate performance report
        perf_report = f"Model Performance Report\n{'='*50}\n\n"
        perf_report += f"Accuracy: {self.accuracy:.4f}\n\n"
        
        # Feature importance
        perf_report += "Feature Importance:\n"
        feature_importance = self.model.feature_importances_
        feature_names = self.X_test.columns
        for name, importance in zip(feature_names, feature_importance):
            perf_report += f"{name}: {importance:.4f}\n"
        
        self.performance_text.setText(perf_report)
        
        # Generate SHAP explanation باستدعاء الدالة الجاهزة فقط
        try:
            # استدعاء دالة التفسير الجاهزة
            shap_values = explain_function(self.model, self.X_test)
            
            # Create summary plot
            self.shap_canvas.figure.clear()
            ax = self.shap_canvas.figure.add_subplot(111)
            shap.summary_plot(shap_values, self.X_test, show=False, plot_type="bar")
            self.shap_canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"SHAP explanation failed: {str(e)}")

    def style(self):
        style = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid grey;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
            }
        """
        return style

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
