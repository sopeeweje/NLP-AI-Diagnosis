from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import json
import numpy

from app import app

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String)
    created_on = db.Column(db.DateTime, default=datetime.utcnow)

    model = db.Column(db.String)
    k = db.Column(db.Integer)
    sizes = db.Column(db.String)
    silhouette_score = db.Column(db.Float)
    train_data_stats = db.Column(db.String)
    validation_data_stats = db.Column(db.String)
    test_data_stats = db.Column(db.String)

    train_accuracy = db.Column(db.Float)
    train_loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    test_loss = db.Column(db.Float)

    probabilities = db.Column(db.String)
    labels = db.Column(db.String)
    test_probabilities = db.Column(db.String)
    test_labels = db.Column(db.String)

    description = db.Column(db.String)
    input_form = db.Column(db.String)
    label = db.Column(db.String)

    hyperparameters = db.Column(db.String)

    history = db.Column(db.String)

    def __repr__(self):
        return '<Result accuracy: {}>'.format(self.accuracy)

    def __init__(self,
            model,
            uuid,
            split_uuid,
            train_data_stats,
            validation_data_stats,
            test_data_stats,
            description,
            input_form,
            label,
            train_accuracy,
            train_loss,
            test_accuracy,
            test_loss,
            accuracy,
            loss,
            probabilities,
            labels,
            test_probabilities,
            test_labels,
            hyperparameters,
            history,
            ):
        self.model = model
        self.uuid = uuid
        self.split_uuid = split_uuid

        self.train_data_stats = json.dumps(train_data_stats, default=default)
        self.validation_data_stats = json.dumps(validation_data_stats, default=default)
        self.test_data_stats = json.dumps(test_data_stats, default=default)

        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.accuracy = accuracy
        self.loss = loss
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss

        self.history = json.dumps(history, default=default)

        self.probabilities = json.dumps(probabilities, default=default)
        self.labels = json.dumps(labels, default=default)

        self.test_probabilities = json.dumps(test_probabilities, default=default)
        self.test_labels = json.dumps(test_labels, default=default)

        self.description = description
        self.input_form = input_form
        self.label = label

        self.hyperparameters = json.dumps(hyperparameters)

    def dict(self):
        return {
            "id": self.id,
            "uuid": self.uuid,
            "model": self.model,
            "createdOn": self.created_on.timestamp(),
            "trainDataStats": json.loads(self.train_data_stats),
            "validationDataStats": json.loads(self.validation_data_stats),
            "trainAccuracy": self.train_accuracy,
            "accuracy": self.accuracy,
            "input_form": self.input_form,
            # is this supposed to be label?
            "label": self.label,
        }

    def results(self):
        return json.loads(self.probabilites), json.loads(self.labels)

    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)

    @property
    def split_seed(self):
        if self.split_uuid:
            return self.split_uuid
        return uuid

    @property
    def label_form(self):
        return self.label