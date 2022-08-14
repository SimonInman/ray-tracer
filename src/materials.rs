use std::sync::Arc;

use rand::rngs::StdRng;
use rand::Rng;

use crate::{
    dot, random_in_unit_sphere, random_unit, random_unit_vector,
    texture::{SolidColour, Texture},
    Colour, HitRecord, Point3, Ray, Vec3,
};

#[derive(Clone)]
pub enum Material {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
    DiffuseLight(DiffuseLight),
}

impl Material {
    pub fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        match &*self {
            Material::Lambertian(lambertian) => lambertian.scatter(ray_in, hit_record),
            Material::Metal(metal) => metal.scatter(ray_in, hit_record),
            Material::Dielectric(dielectric) => dielectric.scatter(ray_in, hit_record),
            Material::DiffuseLight(diffuse_light) => diffuse_light.scatter(ray_in, hit_record),
        }
    }

    pub fn emitted(&self, u: f32, v: f32, point: &Point3) -> Colour {
        match *self {
            Material::DiffuseLight(diffuse_light) => diffuse_light.emitted(u, v, point),
            _ => {
                return Colour {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Lambertian {
    pub(crate) albedo: Arc<dyn Texture + Sync + Send>,
}

// impl Material for Lambertian {
impl Lambertian {
    fn scatter(&self, _ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let mut scatter_direction = hit_record.normal.add(&random_unit_vector());
        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }

        let scattered_ray = Ray {
            origin: hit_record.p,
            direction: scatter_direction,
        };

        let attenuation = self.albedo.value(hit_record.u, hit_record.v, &hit_record.p);

        return Some((scattered_ray, attenuation));
    }

    pub fn new(albedo: Colour) -> Lambertian {
        return Lambertian {
            albedo: Arc::new(SolidColour::new(albedo)),
        };
    }
}

#[derive(Clone, Copy)]
pub struct DiffuseLight {
    pub(crate) colour: Colour,
}
impl DiffuseLight {
    fn scatter(&self, _ray_in: Ray, _hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        return None;
    }

    fn emitted(&self, _u: f32, _v: f32, _point: &Point3) -> Colour {
        return self.colour;
    }
}

#[derive(Clone, Copy)]
pub struct Metal {
    pub(crate) albedo: Colour,
    pub(crate) fuzz: f32,
}

// impl Material for Metal {
impl Metal {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let reflected = reflect(ray_in.direction, hit_record.normal);
        let noise = &random_in_unit_sphere().multiply(self.fuzz);
        let scattered_ray = Ray {
            origin: hit_record.p,
            direction: reflected.add(noise),
        };

        // Hmm, how could this happen? I think maybe if the ray hit the inside
        // of the surface.
        if dot(scattered_ray.direction, hit_record.normal) < 0.0 {
            return None;
        }
        return Some((scattered_ray, self.albedo));
    }
}

#[derive(Clone, Copy)]
pub struct Dielectric {
    pub(crate) index_of_refraction: f32,
}
impl Dielectric {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let attenuation = Colour {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let refraction_ratio = if hit_record.is_front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = ray_in.direction.unit_vector();

        let cos_theta = f32::min(dot(unit_direction.multiply(-1.0), hit_record.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = (refraction_ratio * sin_theta) > 1.0;

        let direction =
            if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_unit() {
                reflect(unit_direction, hit_record.normal)
            } else {
                refract(unit_direction, hit_record.normal, refraction_ratio)
            };

        let scattered_ray = Ray {
            origin: hit_record.p,
            direction,
        };

        return Some((scattered_ray, attenuation));
    }
}

// I didn't really follow the whole derivation of this.
// See section 10.2
fn refract(uv: Vec3, normal: Vec3, eta_over_eta_prime: f32) -> Vec3 {
    let cos_theta = f32::min(dot(uv.multiply(-1.0), normal), 1.0);
    let cos_theta_n = normal.multiply(cos_theta);
    let r_out_perp = cos_theta_n.add(&uv).multiply(eta_over_eta_prime);
    let multiplier = (1.0 - r_out_perp.length_squared()).sqrt();
    let r_out_parallel = normal.multiply(-1.0 * multiplier);
    return r_out_perp.add(&r_out_parallel);
}

// Schlick's approximation for reflectance - this wasn't even explained!
fn reflectance(cosine: f32, refraction_ratio: f32) -> f32 {
    let r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
    let r0_squared = r0 * r0;
    return r0_squared + (1.0 - r0) * (1.0 - cosine).powi(5);
}

// Get the reflection off vector after hitting a surface with unit normal
// surface_normal.
// See https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection
fn reflect(incoming_ray: Vec3, surface_normal: Vec3) -> Vec3 {
    let length_of_b = dot(incoming_ray, surface_normal);

    // Don't really understand the signs of
    // In the diagram the reflection is v + 2B
    // from whence v - 2B?
    // Oh, likely explanation: The dot product gives us a negative value for
    // the length of B. We could alternatively take abs value of this.
    return incoming_ray.subtract(&surface_normal.multiply(2.0 * length_of_b));
}

pub fn random_lambertian(rng: &mut StdRng) -> Lambertian {
    // Sample code does:
    // auto albedo = color::random() * color::random();
    // which is a pointwise multipleication of random colours.
    let mut random_colour_value = || -> f32 { rng.gen_range(0.0..1.0) * rng.gen_range(0.0..1.0) };
    return Lambertian::new(Colour {
        x: random_colour_value(),
        y: random_colour_value(),
        z: random_colour_value(),
    });
}

pub fn random_metal(rng: &mut StdRng) -> Metal {
    let fuzz = rng.gen_range(0.0..0.5);
    return Metal {
        albedo: Colour {
            x: rng.gen_range(0.5..1.0),
            y: rng.gen_range(0.5..1.0),
            z: rng.gen_range(0.5..1.0),
        },
        fuzz: fuzz,
    };
}
