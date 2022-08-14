use crate::{BoundingBox, HitRecord, Material, Point3, Ray};

const EPSILON: f32 = 0.001;

#[derive(Clone)]
pub struct Rectangle {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    z: f32,
    material: Material,
}

impl Rectangle {
    pub fn new(x0: f32, x1: f32, y0: f32, y1: f32, z: f32, material: Material) -> Rectangle {
        return Rectangle {
            x0,
            x1,
            y0,
            y1,
            z,
            material,
        };
    }

    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // first work out what t the ray hits the z plane
        // R(t) = Origin + dir*t
        let t = (self.z - ray.origin.z) / ray.direction.z;
        if t < t_min || t > t_max {
            return None;
        }

        // Calculate x,y hitpoint at t
        let x_hit = ray.origin.x + t * ray.direction.x;
        let y_hit = ray.origin.y + t * ray.direction.y;

        if x_hit < self.x0 || x_hit > self.x1 || y_hit < self.y0 || y_hit > self.y1 {
            return None;
        }

        // It's a hit!
        return Some(HitRecord {
            p: ray.at(t),
            normal: crate::Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            t,
            is_front_face: true,
            material: self.material.clone(),
            u: (x_hit - self.x0) / (self.x1 - self.x0),
            v: (y_hit - self.y0) / (self.y1 - self.y0),
        });
    }

    pub fn bounding_box(&self, _time0: f32, _time1: f32) -> Option<BoundingBox> {
        return Some(BoundingBox {
            minimum: Point3 {
                x: self.x0,
                y: self.y0,
                z: self.z - EPSILON,
            },
            maximum: Point3 {
                x: self.x1,
                y: self.y1,
                z: self.z - EPSILON,
            },
        });
    }
}
