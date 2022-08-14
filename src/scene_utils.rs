use crate::{
    materials::{DiffuseLight, Material},
    Colour, HittableObject, Point3, Sphere,
};

// A position that moves on a rising spiral as the frame increases.
pub fn rising_spiral(frame: usize) -> Point3 {
    let min_radius = 1.0;

    let t = frame as f32 * 0.1;

    // Radius goes from 1 to 3 as t goes from 0 to 4.
    // For rotation, one spin is going from zero to 2pi,
    // so going from 0 to 8 seems reasonable.
    let x = (min_radius + (t * 0.5)) * (t * 2.0).sin();
    let z = (min_radius + (t * 0.5)) * (t * 2.0).cos();

    // Height moves from 0 to 2 over 40 frames.
    let y = t * 0.5;

    return Point3 { x, y, z };
}

pub(crate) fn circling_lights(
    frame: usize,
    num_lights: usize,
    radius: f32,
    height: f32,
) -> Vec<HittableObject<'static>> {
    let t = frame as f32 * 0.1;

    let light = DiffuseLight {
        colour: Colour {
            x: 4.0,
            y: 4.0,
            z: 0.8,
        },
    };

    let mut world_list: Vec<HittableObject> = Vec::new();
    let pi = std::f32::consts::PI;

    for light_num in 0..num_lights {
        // Distribute lights evenly between 0 and 2pi

        let this_light_offset = (light_num as f32 / (num_lights as f32)) * 2.0 * pi;
        let location = Point3 {
            x: -radius * ((t + this_light_offset) * 0.5).sin(),
            y: height,
            z: -radius * ((t + this_light_offset) * 0.5).cos(),
        };
        world_list.push(HittableObject::Sphere(Sphere {
            centre: location,
            radius: 0.18,
            material: Material::DiffuseLight(light),
        }));
    }
    return world_list;
}
