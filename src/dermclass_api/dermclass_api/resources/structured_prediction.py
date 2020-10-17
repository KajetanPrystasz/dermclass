from flask_restful import Resource
from flask import request
from dermclass_api.models.structured_prediction import StructuredPredictionModel, StructuredPredictionSchema
from dermclass_structured.predict import make_prediction
from dermclass_api import logger as _logger


class StructuredPrediction(Resource):
    schema = StructuredPredictionSchema()

    def get(self, prediction_id):
        prediction = StructuredPredictionModel.find_by_prediction_id(prediction_id)
        if prediction:
            return prediction.json(), 200
        return {'message': 'prediction not found'}, 404

    def post(self, prediction_id):
        if StructuredPredictionModel.find_by_prediction_id(prediction_id):
            return {'message': f"An prediction with id '{prediction_id}' already exists."}, 400

        data = request.get_json()
        data_valid = StructuredPrediction.schema.load(data)
        data_valid["target"] = make_prediction(data_valid).tolist()[0]
        _logger.debug(f'Outputs: {data_valid["target"]}')
        prediction = StructuredPredictionModel(prediction_id, **data_valid)

        try:
            prediction.save_to_db()
        except:
            return {"message": "An error occurred inserting the item."}, 500

        return prediction.json(), 201

    def delete(self, prediction_id):
        prediction = StructuredPredictionModel.find_by_prediction_id(prediction_id)
        if prediction:
            prediction.delete_from_db()
            return {'message': 'Prediction deleted.'}, 200
        return {'message': 'Prediction not found.'}, 404
