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
            return prediction.json()
        return {'message': 'prediction not found'}, 404

    def post(self, prediction_id):
        if StructuredPredictionModel.find_by_prediction_id(prediction_id):
            return {'message': f"An prediction with id '{prediction_id}' already exists."}, 400

        data = request.get_json()
        data_valid = StructuredPrediction.schema.load(data)

        data_valid["target_"] = make_prediction(data_valid)
        _logger.debug(f'Outputs: {data_valid["target_"]}')
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
            return {'message': 'Prediction deleted.'}
        return {'message': 'Prediction not found.'}, 404


#@prediction_app.route('/v1/predict/regression', methods=['POST'])
#def predict():
#    if request.method == 'POST':
#        # Step 1: Extract POST data from request body as JSON
#        json_data = request.get_json()
#        _logger.debug(f'Inputs: {json_data}')
#
#        # Step 2: Validate the input using marshmallow schema
#        input_data, errors = validate_inputs(input_data=json_data)
#
#        # Step 3: Model prediction
#        result = make_prediction(input_data=input_data)
#        _logger.debug(f'Outputs: {result}')
#
#        # Step 4: Convert numpy ndarray to list
#        predictions = result.get('predictions').tolist()
#        version = result.get('version')
#
#        # Step 5: Return the response as JSON
#        return jsonify({'predictions': predictions,
#                        'version': version,
#                        'errors': errors})
