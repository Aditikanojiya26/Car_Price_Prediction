from pydantic import BaseModel


class CarFeatures(BaseModel):
    brand: str
    car_name: str
    manufacture: int
    kms_driven: float
    fuel_type: str
    transmission: str
    ownership: int
    engine: float
    Seats: int
    car_age: int