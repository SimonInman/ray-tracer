use std::sync::Arc;

use crate::{Point3, Colour};

pub trait Texture {
    fn value(&self, u: f32, v: f32, hit_point: &Point3 ) -> Colour;
}

#[derive(Clone, Copy)]
pub struct SolidColour {
    colour_value: Colour, 
}

impl SolidColour {
    pub(crate) fn new(colour: Colour) -> SolidColour {
        return SolidColour { colour_value: colour};
    }
}

impl Texture for SolidColour {
    fn value(&self, _u: f32, _v: f32, _hit_point: &Point3 ) -> Colour {
        return self.colour_value;
    }
}

#[derive(Clone)]
pub struct CheckerTexture {
    even: Arc<dyn Texture + Sync + Send>,
    odd: Arc<dyn Texture + Sync + Send>,
}

impl Texture for CheckerTexture {
    fn value(&self,u: f32, v: f32, hit_point: &Point3 ) -> Colour {
    let sines = (10.0*hit_point.x).sin() * 
    (10.0*hit_point.y).sin() * 
    (10.0*hit_point.z).sin();
    if sines < 0.0 { 
        return self.even.value(u, v, hit_point);
    } else {
        return self.odd.value(u, v, hit_point);

    }

    }

}

impl CheckerTexture {
    pub fn new(c1: Colour, c2: Colour) -> CheckerTexture {
        return CheckerTexture { 
            even: Arc::new(SolidColour{colour_value: c1}), 
            odd: Arc::new( SolidColour{colour_value: c2}), 
        };

    }
}