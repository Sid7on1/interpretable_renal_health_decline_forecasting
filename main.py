import logging
import os
import sys
from typing import Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from chain_of_thought import ChainOfThought
from abductive_reasoning import AbductiveReasoning
from knowledge_transfer import KnowledgeTransfer
from short_term_memory import ShortTermMemory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self):
        self.data_path = 'data.csv'
        self.model_name = 'bert-base-uncased'
        self.batch_size = 32
        self.epochs = 5
        self.learning_rate = 1e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Main:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = config.device

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            data = pd.read_csv(self.config.data_path)
            X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            self.logger.error(f'File not found: {self.config.data_path}')
            sys.exit(1)

    def initialize_models(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            return model, tokenizer
        except Exception as e:
            self.logger.error(f'Failed to initialize models: {e}')
            sys.exit(1)

    def generate_predictions(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, X_test: pd.Series) -> np.ndarray:
        try:
            inputs = tokenizer(X_test, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            return predictions
        except Exception as e:
            self.logger.error(f'Failed to generate predictions: {e}')
            sys.exit(1)

    def generate_explanations(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, X_test: pd.Series) -> List[Dict]:
        try:
            chain_of_thought = ChainOfThought(model, tokenizer)
            abductive_reasoning = AbductiveReasoning(model, tokenizer)
            knowledge_transfer = KnowledgeTransfer(model, tokenizer)
            short_term_memory = ShortTermMemory(model, tokenizer)
            explanations = []
            for text in X_test:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                explanation = chain_of_thought.explain(outputs.logits)
                explanation = abductive_reasoning.explain(explanation)
                explanation = knowledge_transfer.explain(explanation)
                explanation = short_term_memory.explain(explanation)
                explanations.append(explanation)
            return explanations
        except Exception as e:
            self.logger.error(f'Failed to generate explanations: {e}')
            sys.exit(1)

    def train_model(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, X_train: pd.Series, y_train: pd.Series) -> None:
        try:
            train_dataset = self.create_dataset(X_train, y_train, tokenizer)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            for epoch in range(self.config.epochs):
                model.train()
                for batch in train_loader:
                    inputs = {k: v.to(self.device) for k, v in batch}
                    outputs = model(**inputs)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                predictions = self.generate_predictions(model, tokenizer, X_train)
                accuracy = accuracy_score(y_train, predictions)
                self.logger.info(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}')
        except Exception as e:
            self.logger.error(f'Failed to train model: {e}')
            sys.exit(1)

    def create_dataset(self, X: pd.Series, y: pd.Series, tokenizer: AutoTokenizer) -> torch.utils.data.Dataset:
        try:
            inputs = tokenizer(X, return_tensors='pt', padding=True, truncation=True)
            labels = torch.tensor(y, dtype=torch.long)
            return torch.utils.data.Dataset.from_tensor_data_list((inputs, labels))
        except Exception as e:
            self.logger.error(f'Failed to create dataset: {e}')
            sys.exit(1)

    def run(self) -> None:
        try:
            X_train, X_test, y_train, y_test = self.load_data()
            model, tokenizer = self.initialize_models()
            self.train_model(model, tokenizer, X_train, y_train)
            predictions = self.generate_predictions(model, tokenizer, X_test)
            explanations = self.generate_explanations(model, tokenizer, X_test)
            accuracy = accuracy_score(y_test, predictions)
            self.logger.info(f'Test Accuracy: {accuracy:.4f}')
            self.logger.info('Classification Report:')
            self.logger.info(classification_report(y_test, predictions))
            self.logger.info('Confusion Matrix:')
            self.logger.info(confusion_matrix(y_test, predictions))
        except Exception as e:
            self.logger.error(f'Failed to run application: {e}')
            sys.exit(1)

if __name__ == '__main__':
    config = Config()
    main = Main(config)
    main.run()