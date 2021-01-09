import logging
from typing import Tuple

from PIL import Image
import numpy as np

from flask import request, flash
from flask_restful import Resource

from dermclass_models.prediction import StructuredPrediction, TextPrediction, ImagePrediction

from dermclass_api.prediction_models import (StructuredPredictionModel, StructuredPredictionSchema,
                                             TextPredictionModel, TextPredictionSchema,
                                             ImagePredictionModel, ImagePredictionSchema)

logger = logging.getLogger(__name__)


# TODO: Refactor class to be abstract. For now it crashes because of use of SQLAlchemy
class _BasePrediction:

    def __init__(self, schema, model, prediction_obj):
        """
        Base prediction resource class for flask-restful. The class implement basic endpoints: get, post, delete.
        param schema: A Marshmallow schema for validation purposes
        param model: A SQLalchemy prediction model for persistence
        param prediction_obj: A prediction object from dermclass_model
        """
        self.schema = schema
        self.model = model
        self.prediction_obj = prediction_obj

    def get(self, prediction_id: int) -> Tuple[dict, int]:
        """
        Get endpoint for the data. The function returns a prediction with provided prediction_id from the db
        param prediction_id: A prediction id to find the prediction in db
        return: Returns a tuple of message output information and HTTP code
        """
        prediction = self.model.find_by_prediction_id(prediction_id)
        if prediction:
            return prediction.json(), 200

        return_message = {'message': 'prediction not found'}, 404
        logger.info(return_message)
        return return_message

    def post(self, prediction_id: int) -> Tuple[dict, int]:
        """
        Post endpoint for the data. The function uses ML models to make prediction on imputed data and save to db.
        param prediction_id: A prediction id to save the prediction in db with
        return: Returns a tuple of message output information and HTTP code
        """
        if self.model.find_by_prediction_id(prediction_id):
            return {'message': f"An prediction with id '{prediction_id}' already exists."}, 400

        data = request.get_json()
        data_valid = self.schema.load(data=data)
        data_valid["prediction_proba"], data_valid["prediction_string"] = self.prediction_obj.make_prediction(data_valid)

        logger.debug(f'Outputs: {data_valid["prediction_proba"]}, {data_valid["prediction_string"]}')
        prediction = self.model(prediction_id, **data_valid)

        try:
            logger.info("Saving prediction to db")
            prediction.save_to_db()
        except:
            return_message = {"message": "An error occurred inserting the item."}, 500
            logger.info(return_message)
            return return_message

    def delete(self, prediction_id: int) -> Tuple[dict, int]:
        """
        Delete endpoint for the data. The function deletes given prediction from the database
        param prediction_id: A prediction id to find the prediction in db
        return: Returns a tuple of message output information and HTTP code
        """
        prediction = self.model.find_by_prediction_id(prediction_id)
        if prediction:
            prediction.delete_from_db()
            return_message = {'message': 'Prediction deleted.'}, 200
            logger.info(return_message)
            return return_message
        return_message = {'message': 'Prediction not found.'}, 404
        logger.info(return_message)
        return return_message


class StructuredPredictionResource(_BasePrediction, Resource):
    def __init__(self,
                 schema=StructuredPredictionSchema(),
                 model=StructuredPredictionModel,
                 prediction_obj=StructuredPrediction()):
        """Structured prediction resource, for more documentation lookup into _BasePrediction documentation"""
        super().__init__(schema, model, prediction_obj)


class TextPredictionResource(_BasePrediction, Resource):
    def __init__(self,
                 schema=TextPredictionSchema(),
                 model=TextPredictionModel,
                 prediction_obj=TextPrediction()):
        """Text prediction resource, for more documentation lookup into _BasePrediction documentation"""

        super().__init__(schema, model, prediction_obj)


class ImagePredictionResource(_BasePrediction, Resource):
    def __init__(self,
                 schema=ImagePredictionSchema(),
                 model=ImagePredictionModel,
                 prediction_obj=ImagePrediction()):
        """Image prediction resource, for more documentation lookup into _BasePrediction documentation"""
        super().__init__(schema, model, prediction_obj)
        self.id_counter = 0

    # TODO: Add saving to persistent file storage -> refactor this function
    def post(self, prediction_id: int) -> Tuple[dict, int]:
        """
        Post endpoint for the data. The function uses ML models to make prediction on imputed data and save to db.
        WARNING: For now the function is in development mode and does not implement saving to persistent storage
        param prediction_id: A prediction id to save the prediction in db with
        return: Returns a tuple of message output information and HTTP code
        """
        if self.model.find_by_prediction_id(prediction_id):
            return {'message': f"A prediction with id '{prediction_id}' already exists."}, 400
        if 'file' not in request.files:
            flash('No file inputted')
            raise ValueError("No file inputted")

        img_file = request.files['file']
        with open(f"temp/img_file_{self.id_counter}.jpeg", "wb") as f:
            self.id_counter += 1
            f.write(img_file.read())

        data = {}
        data_valid = self.schema.load(data)

        img = Image.open(img_file)
        data_valid["img_array"] = np.array(img)

        data_valid["prediction_proba"], data_valid["prediction_string"] = self.prediction_obj.make_prediction(data_valid)
        logger.debug(f'Outputs: {data_valid["prediction_proba"]}, {data_valid["prediction_string"]}')

        data_valid.pop("img_array")
        prediction = self.model(prediction_id, **data_valid)

        try:
            prediction.save_to_db()
        except:
            return {"message": "An error occurred inserting the item."}, 500
