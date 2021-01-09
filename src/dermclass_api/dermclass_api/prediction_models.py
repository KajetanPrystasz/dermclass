import logging
from typing import List

from dermclass_api.extensions import db, ma
from dermclass_models import __version__ as model_version

logger = logging.getLogger(__name__)


class _PredictionModel:
    prediction_id = db.Column(db.Integer, primary_key=True)
    prediction_proba = None
    prediction_string = None

    def json(self) -> dict:
        """
        Basic function to get basic info to jsonify
        return: Returns a dict version of prediction information
        """
        return {"prediction_id": self.prediction_id,
                "prediction_proba": self.prediction_proba,
                "prediction_string": self.prediction_string,
                "version": model_version}

    @classmethod
    def find_by_prediction_id(cls, prediction_id: int) -> List:
        prediction = cls.query.filter_by(prediction_id=prediction_id).first()
        if prediction:
            logger.info(f"Item with {prediction_id} prediction_id found in database")
            return prediction
        else:
            logger.info(f"Item with {prediction_id} prediction_id not found in database")

    def save_to_db(self):
        """ Utility function to save given item to database"""
        db.session.add(self)
        db.session.commit()
        logger.info("Item properly saved to the database")

    def delete_from_db(self):
        """Utility function to delete given item from database"""
        db.session.delete(self)
        db.session.commit()
        logger.info("Item properly deleted from the database")


class StructuredPredictionModel(_PredictionModel, db.Model):

    __tablename__ = 'structuredPredictions'

    erythema = db.Column(db.Integer, nullable=False)
    scaling = db.Column(db.Integer, nullable=False)
    definite_borders = db.Column(db.Integer, nullable=False)
    itching = db.Column(db.Integer, nullable=False)
    koebner_phenomenon = db.Column(db.Integer, nullable=False)
    polygonal_papules = db.Column(db.Integer, nullable=False)
    follicular_papules = db.Column(db.Integer, nullable=False)
    oral_mucosal_involvement = db.Column(db.Integer, nullable=False)
    knee_and_elbow_involvement = db.Column(db.Integer, nullable=False)
    scalp_involvement = db.Column(db.Integer, nullable=False)
    family_history = db.Column(db.Integer, nullable=False)
    melanin_incontinence = db.Column(db.Integer, nullable=False)
    eosinophils_in_the_infiltrate = db.Column(db.Integer, nullable=False)
    pnl_infiltrate = db.Column(db.Integer, nullable=False)
    fibrosis_of_the_papillary_dermis = db.Column(db.Integer, nullable=False)
    exocytosis = db.Column(db.Integer, nullable=False)
    acanthosis = db.Column(db.Integer, nullable=False)
    hyperkeratosis = db.Column(db.Integer, nullable=False)
    parakeratosis = db.Column(db.Integer, nullable=False)
    clubbing_of_the_rete_ridges = db.Column(db.Integer, nullable=False)
    elongation_of_the_rete_ridges = db.Column(db.Integer, nullable=False)
    thinning_of_the_suprapapillary_epidermis = db.Column(db.Integer, nullable=False)
    spongiform_pustule = db.Column(db.Integer, nullable=False)
    munro_microabcess = db.Column(db.Integer, nullable=False)
    focal_hypergranulosis = db.Column(db.Integer, nullable=False)
    disappearance_of_the_granular_layer = db.Column(db.Integer, nullable=False)
    vacuolisation_and_damage_of_basal_layer = db.Column(db.Integer, nullable=False)
    spongiosis = db.Column(db.Integer, nullable=False)
    saw_tooth_appearance_of_retes = db.Column(db.Integer, nullable=False)
    follicular_horn_plug = db.Column(db.Integer, nullable=False)
    perifollicular_parakeratosis = db.Column(db.Integer, nullable=False)
    inflammatory_monoluclear_inflitrate = db.Column(db.Integer, nullable=False)
    band_like_infiltrate = db.Column(db.Integer, nullable=False)

    age = db.Column(db.Integer, nullable=False)

    prediction_proba = db.Column(db.Float, nullable=False)
    prediction_string = db.Column(db.String, nullable=False)

    def __init__(self, prediction_id, erythema, scaling, definite_borders, itching, koebner_phenomenon,
                 polygonal_papules, follicular_papules, oral_mucosal_involvement, knee_and_elbow_involvement,
                 scalp_involvement, family_history, melanin_incontinence, eosinophils_in_the_infiltrate, pnl_infiltrate,
                 fibrosis_of_the_papillary_dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis,
                 clubbing_of_the_rete_ridges, elongation_of_the_rete_ridges, thinning_of_the_suprapapillary_epidermis,
                 spongiform_pustule, munro_microabcess, focal_hypergranulosis, disappearance_of_the_granular_layer,
                 vacuolisation_and_damage_of_basal_layer, spongiosis, saw_tooth_appearance_of_retes,
                 follicular_horn_plug, perifollicular_parakeratosis, inflammatory_monoluclear_inflitrate,
                 band_like_infiltrate, age, prediction_proba, prediction_string):
        """Structured model for the SQL alchemy database"""

        self.prediction_id = prediction_id

        self.erythema = erythema
        self.scaling = scaling
        self.definite_borders = definite_borders
        self.itching = itching
        self.koebner_phenomenon = koebner_phenomenon
        self.polygonal_papules = polygonal_papules
        self.follicular_papules = follicular_papules
        self.oral_mucosal_involvement = oral_mucosal_involvement
        self.knee_and_elbow_involvement = knee_and_elbow_involvement
        self.scalp_involvement = scalp_involvement
        self.family_history = family_history
        self.melanin_incontinence = melanin_incontinence
        self.eosinophils_in_the_infiltrate = eosinophils_in_the_infiltrate
        self.pnl_infiltrate = pnl_infiltrate
        self.fibrosis_of_the_papillary_dermis = fibrosis_of_the_papillary_dermis
        self.exocytosis = exocytosis
        self.acanthosis = acanthosis
        self.hyperkeratosis = hyperkeratosis
        self.parakeratosis = parakeratosis
        self.clubbing_of_the_rete_ridges = clubbing_of_the_rete_ridges
        self.elongation_of_the_rete_ridges = elongation_of_the_rete_ridges
        self.thinning_of_the_suprapapillary_epidermis = thinning_of_the_suprapapillary_epidermis
        self.spongiform_pustule = spongiform_pustule
        self.munro_microabcess = munro_microabcess
        self.focal_hypergranulosis = focal_hypergranulosis
        self.disappearance_of_the_granular_layer = disappearance_of_the_granular_layer
        self.vacuolisation_and_damage_of_basal_layer = vacuolisation_and_damage_of_basal_layer
        self.spongiosis = spongiosis
        self.saw_tooth_appearance_of_retes = saw_tooth_appearance_of_retes
        self.follicular_horn_plug = follicular_horn_plug
        self.perifollicular_parakeratosis = perifollicular_parakeratosis
        self.inflammatory_monoluclear_inflitrate = inflammatory_monoluclear_inflitrate
        self.band_like_infiltrate = band_like_infiltrate

        self.age = age

        self.prediction_proba = prediction_proba
        self.prediction_string = prediction_string


class TextPredictionModel(_PredictionModel, db.Model):

    __tablename__ = 'textPredictions'

    text = db.Column(db.String, nullable=False)
    prediction_proba = db.Column(db.Float, nullable=False)
    prediction_string = db.Column(db.String, nullable=False)

    def __init__(self, prediction_id, text, prediction_proba, prediction_string):
        """Text model for the SQL alchemy database"""
        self.prediction_id = prediction_id
        self.text = text
        self.prediction_proba = prediction_proba
        self.prediction_string = prediction_string


class ImagePredictionModel(_PredictionModel, db.Model):

    __tablename__ = 'ImagePredictions'

    prediction_proba = db.Column(db.Float, nullable=False)
    prediction_string = db.Column(db.String, nullable=False)

    def __init__(self, prediction_id, prediction_proba, prediction_string):
        """Image model for the SQL alchemy database"""

        self.prediction_id = prediction_id
        self.prediction_proba = prediction_proba
        self.prediction_string = prediction_string


class StructuredPredictionSchema(ma.SQLAlchemyAutoSchema):
    """Automatic Structured table, Marshmallow-SQLAlchemy schema for validation"""
    class Meta:
        model = StructuredPredictionModel
        exclude = ("prediction_id", "prediction_proba", "prediction_string")


class TextPredictionSchema(ma.SQLAlchemyAutoSchema):
    """Automatic Text table, Marshmallow-SQLAlchemy schema for validation"""
    class Meta:
        model = TextPredictionModel
        exclude = ("prediction_id", "prediction_proba", "prediction_string")


class ImagePredictionSchema(ma.SQLAlchemyAutoSchema):
    """Automatic Image table, Marshmallow-SQLAlchemy schema for validation"""
    class Meta:
        model = ImagePredictionModel
        exclude = ("prediction_id", "prediction_proba", "prediction_string")
