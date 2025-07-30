from pydantic import BaseModel


class CameraData(BaseModel):
    x:  float
    y:  float
    z:  float


class AttributeData(BaseModel):
    camera_position:    CameraData
    camera_rotation:    CameraData
    
    iteration:          int

    screenshot1:        str
    screenshot2:        str


class RandomAttributeData(AttributeData):
    high_poyl_in_frame: bool
