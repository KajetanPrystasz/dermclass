import logging

from PIL import Image
import numpy as np

from flask_restful import Resource
from flask import request, flash

from dermclass_models.prediction import StructuredPrediction, TextPrediction, ImagePrediction

from dermclass_api.prediction_models import (StructuredPredictionModel, StructuredPredictionSchema,
                                             TextPredictionModel, TextPredictionSchema,
                                             ImagePredictionModel, ImagePredictionSchema)

logger = logging.getLogger(__name__)


class BasePrediction:

    def __init__(self, schema, model, prediction_obj):
        self.schema = schema
        self.model = model
        self.prediction_obj = prediction_obj

    def get(self, prediction_id):
        prediction = self.model.find_by_prediction_id(prediction_id)
        if prediction:
            return prediction.json(), 200
        return {'message': 'prediction not found'}, 404

    def post(self, prediction_id):
        if self.model.find_by_prediction_id(prediction_id):
            return {'message': f"An prediction with id '{prediction_id}' already exists."}, 400

        data = request.get_json()

        data_valid = self.schema.load(data)
        data_valid["target"] = self.prediction_obj(data_valid)
        logger.debug(f'Outputs: {data_valid["target"]}')
        prediction = self.model(prediction_id, **data_valid)

        try:
            prediction.save_to_db()
        except:
            return {"message": "An error occurred inserting the item."}, 500

    def delete(self, prediction_id):
        prediction = self.model.find_by_prediction_id(prediction_id)
        if prediction:
            prediction.delete_from_db()
            return {'message': 'Prediction deleted.'}, 200
        return {'message': 'Prediction not found.'}, 404


class StructuredPredictionResource(BasePrediction, Resource):
    def __init__(self, schema=StructuredPredictionSchema,
                 model=StructuredPredictionModel,
                 prediction_obj=StructuredPrediction()):
        super().__init__(schema, model, prediction_obj)


class TextPredictionResource(BasePrediction, Resource):
    def __init__(self, schema=TextPredictionSchema,
                 model=TextPredictionModel,
                 prediction_obj=TextPrediction()):
        super().__init__(schema, model, prediction_obj)


class ImagePredictionResource(BasePrediction, Resource):
    def __init__(self, schema=ImagePredictionSchema,
                 model=ImagePredictionModel,
                 prediction_obj=ImagePrediction()):
        super().__init__(schema, model, prediction_obj)

    def post(self, prediction_id):
        if self.model.find_by_prediction_id(prediction_id):
            return {'message': f"An prediction with id '{prediction_id}' already exists."}, 400

        if 'file' not in request.files:
            flash('No file inputted')
            raise ValueError("No file inputted")

        data = request.get_json()

        image_file = request.files['file']
        image = Image.open(image_file)
        np_image = np.array(image)
        data_with_img = data.update({"imgArray": np_image})

        data_valid = self.schema.load(data_with_img)

        data_valid["target"] = self.prediction_obj(data_valid).make_prediction(input_data=data_valid)
        logger.debug(f'Outputs: {data_valid["target"]}')

        prediction = self.model(prediction_id, **data_valid)

        try:
            prediction.save_to_db()
        except:
            return {"message": "An error occurred inserting the item."}, 500
