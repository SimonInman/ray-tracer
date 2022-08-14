use crate::{Point3};

// A position that moves on a rising spiral as the frame increases.
pub fn rising_spiral(frame: usize) -> Point3 {

    let min_radius = 1.0;

    let t = frame as f32 * 0.1;

    // Radius goes from 1 to 3 as t goes from 0 to 4.
    // For rotation, one spin is going from zero to 2pi, 
    // so going from 0 to 8 seems reasonable.
    let x = (min_radius + (t*0.5)) * (t*2.0).sin();
    let z = (min_radius + (t*0.5)) * (t*2.0).cos();

    // Height moves from 0 to 2 over 40 frames.
    let y = t * 0.5;

    return Point3 { x, y, z}
}