pub mod materials;
pub mod rectangle;
pub mod scene_utils;
pub mod texture;

use materials::random_lambertian;
use materials::random_metal;
use materials::Dielectric;
use materials::DiffuseLight;
use materials::Lambertian;
use materials::Material;
use materials::Metal;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use rectangle::Rectangle;
use scene_utils::circling_lights;
use scene_utils::rising_spiral;
use std::cmp::Ordering;
use std::fs::File;
use std::io::Write;
use texture::CheckerTexture;

use std::sync::Arc;

const MAX_T: f32 = 20000.0;
const OUTPUT_DIR: &'static str = "output/gif7/";
const FRAME_PREFIX: &'static str = "frame_";
const FRAME_SUFFIX: &'static str = ".ppm";
const NUM_FRAMES: usize = 50;

fn main() {
    for frame in 0..NUM_FRAMES {
        let padded_frame = format!("{:0>2}", frame);
        let filepath = format!(
            "{}{}{}{}",
            OUTPUT_DIR, FRAME_PREFIX, padded_frame, FRAME_SUFFIX
        );
        let file = File::create(filepath).expect("file creation failed");

        let mut build_spheres = many_spheres(frame);
        let world_bvh = BVH::build_bvh(&mut build_spheres, 0.0, 0.0);
        let boxed_world = HittableObject::BVH(world_bvh);

        let light_z_axis_start = 1.0;
        let frame_ticker = light_z_axis_start + 0.1 * frame as f32;

        println!("Rendering frame {}...", padded_frame);
        render_frame(frame_ticker, boxed_world, file);
    }
}

fn render_frame(frame_ticker: f32, boxed_world: HittableObject, mut file: File) {
    // Image
    let aspect_ratio = 3.0 / 2.0;
    let image_width = 600;
    // let image_width = 1200;
    let image_height: i32 = (image_width as f32 / aspect_ratio) as i32;
    let samples_per_pixel = 50;
    // let samples_per_pixel = 500;
    let max_depth = 50;

    let look_from = Point3 {
        x: 13.0 + 1.0 - frame_ticker * 1.0,
        y: 5.0 - 0.5 + frame_ticker * 0.5,
        z: 7.0 - 2.3 + frame_ticker * 2.3,
    };
    let look_at = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let v_up = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let camera = build_camera(
        look_from,
        look_at,
        v_up,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    // let background = Colour {
    //     x: 0.70,
    //     y: 0.80,
    //     z: 1.00,
    // };
    let grey_background = Colour {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    // Render
    writeln!(file, "P3").unwrap();
    writeln!(file, "{}", image_width.to_string()).unwrap();
    writeln!(file, "{}", image_height.to_string()).unwrap();
    writeln!(file, "255").unwrap();

    // See https://raytracing.github.io/images/fig-1.03-cam-geom.jpg
    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let aggregated_pixel = (0..samples_per_pixel)
                .into_par_iter()
                .map(|_sample| {
                    let u = (i as f32 + random_unit()) / (image_width as f32 - 1.0); // why minus one?
                    let v = (j as f32 + random_unit()) / (image_height as f32 - 1.0); // why minus one?
                    let ray = camera.get_ray(u, v);
                    return ray_colour(ray, &boxed_world, max_depth, grey_background);
                })
                .reduce(
                    || Colour {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    |accum, color| accum.add(&color),
                );

            write_pixel(aggregated_pixel, samples_per_pixel, &mut file);
        }
    }
}

fn many_spheres(frame: usize) -> Vec<HittableObject<'static>> {
    let light = DiffuseLight {
        colour: Colour {
            x: 4.0,
            y: 4.0,
            z: 0.8,
        },
    };
    let material_ground = Lambertian::new(Colour {
        x: 0.5,
        y: 0.5,
        z: 0.5,
    });
    let checker_texture = CheckerTexture::new(
        Colour {
            x: 0.8,
            y: 0.5,
            z: 0.1,
        },
        Colour {
            x: 0.1,
            y: 0.2,
            z: 0.8,
        },
    );
    let material_lambertian = Lambertian {
        albedo: Arc::new(checker_texture),
    };
    let material_glass = Dielectric {
        index_of_refraction: 1.5,
    };

    let mut world_list: Vec<HittableObject> = Vec::new();
    // Add "ground" sphere.
    world_list.push(HittableObject::Sphere(Sphere {
        centre: Vec3 {
            x: 0.0,
            y: -1000.0,
            z: 0.0,
        },
        radius: 1000.0,
        material: Material::Lambertian(material_ground),
    }));

    let mut seeded_rng = StdRng::seed_from_u64(230);
    let world_size = 8;

    for a in -world_size..world_size {
        for b in -world_size..world_size {
            let centre = Vec3 {
                // x: random_unit() * 0.9 + a as f32,
                x: seeded_rng.gen_range(0.0..1.0) * 0.9 + a as f32,
                y: 0.2,
                // z: random_unit() * 0.9 + b as f32,
                z: seeded_rng.gen_range(0.0..1.0) * 0.9 + b as f32,
            };

            let threshold_point = Point3 {
                x: 4.0,
                y: 0.2,
                z: 0.0,
            };

            let noise = random_unit();
            // Don't render if we're too near?
            if centre.subtract(&threshold_point).length() > 0.9 {
                if noise < 0.8 {
                    // Diffuse material
                    world_list.push(HittableObject::Sphere(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Lambertian(random_lambertian(&mut seeded_rng)),
                    }));
                } else if noise < 0.95 {
                    // Metal
                    world_list.push(HittableObject::Sphere(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Metal(random_metal(&mut seeded_rng)),
                    }));
                } else {
                    //glass
                    world_list.push(HittableObject::Sphere(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Dielectric(material_glass),
                    }));
                }
            }
        }
    }

    world_list.push(HittableObject::Sphere(Sphere {
        centre: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.0,
        material: Material::Dielectric(material_glass),
    }));
    world_list.push(HittableObject::Sphere(Sphere {
        centre: Vec3 {
            x: -4.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.0,
        material: Material::Lambertian(material_lambertian),
    }));
    let material_metal = Metal {
        albedo: Colour {
            x: 0.7,
            y: 0.6,
            z: 0.5,
        },
        fuzz: 0.0,
    };
    world_list.push(HittableObject::Sphere(Sphere {
        centre: Vec3 {
            x: 4.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.0,
        material: Material::Metal(material_metal),
    }));

    world_list.push(HittableObject::Sphere(Sphere {
        centre: rising_spiral(frame),
        // Radius grows over first 10 frames
        radius: 0.25 * (frame as f32 * 0.1).min(0.25),
        material: Material::DiffuseLight(light),
    }));
    if frame > 10 {
        let delayed_frame = frame - 10;
        world_list.push(HittableObject::Sphere(Sphere {
            centre: rising_spiral(delayed_frame),
            // Radius grows over first 10 frames
            radius: 0.25 * (delayed_frame as f32 * 0.1).min(0.25),
            material: Material::DiffuseLight(light),
        }));
    }

    let mut light_list = circling_lights(frame, 11, 5.0, 2.2);

    world_list.append(&mut light_list);

    return world_list;
}

fn write_pixel(colour: Colour, samples_per_pixel: i32, file: &mut File) {
    let scale = 1.0 / (samples_per_pixel as f32);
    // Sqrt for gamma correction.
    let r = (colour.x * scale).sqrt();
    let g = (colour.y * scale).sqrt();
    let b = (colour.z * scale).sqrt();

    let out = [
        safe_colour_print(r),
        safe_colour_print(g),
        safe_colour_print(b),
    ]
    .join(" ");
    writeln!(file, "{}", out).unwrap();
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
fn cross(u: Vec3, v: Vec3) -> Vec3 {
    return Vec3 {
        x: (u.y * v.z) - u.z * v.y,
        y: (u.z * v.x) - u.x * v.z,
        z: (u.x * v.y) - u.y * v.x,
    };
}

// You can form a vector quadratic equation using the extension of the Ray (which is itself a function of time, t),
// This gives us an equation for whether there is any T such that the ray strikes the sphere. This quadratic has
// either 0, 1 or 2 solutions, corresponding to mising, clipping, or going through the sphere
//
// The a, b, c components of the quadratic are derived here:
// https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/creatingourfirstraytracedimage
// (The half_b optimisation is described in section 6.2.)
//
// Returns the time t at which the ray first hit the sphere.
// fn hit_sphere(centre: Vec3, radius: f32, ray: &Ray) -> Option<f32> {
//     let origin_centre = ray.origin.subtract(&centre);
//     let a = dot(ray.direction, ray.direction);
//     let half_b = dot(origin_centre, ray.direction);
//     // let b = 2.0 * dot(origin_centre, ray.direction);
//     let c = dot(origin_centre, origin_centre) - radius * radius;
//     let discriminant = half_b * half_b - a * c;
//     // return discriminant > 0.0;
//     if discriminant < 0.0 {
//         return None;
//     } else {
//         // Returning first hit, which is lowest t, so -b - sqrt(disc)/2a
//         return Some((-half_b - discriminant.sqrt()) / a);
//     }
// }

/// Colour a ray depending on if it hits a sphere at the centre of our viewport.
/// If not hit (i.e. time_hit is None), display background gradient as in commented
// out ray_colour.
// If it does hit, colour depending on the direction of the normal at the point
// of impact.
// The unit normal is given by the point of contact, P, minus the sphere centre.
// (If you're standing at P on earth, the direction to the centre of the earth is
// C - P, so the opposite direction is P-C).
// The chose colour is to take the unit normal and use it's parameters as colours.
fn ray_colour(ray: Ray, world: &HittableObject, depth: i32, background: Colour) -> Colour {
    if depth <= 0 {
        return Colour {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    // Did we hit anything?
    let maybe_hit_record = world.hit(&ray, 0.001, MAX_T);
    if maybe_hit_record.is_none() {
        return background;
    }
    let hit_record = maybe_hit_record.unwrap();

    let emitted = hit_record
        .material
        .emitted(hit_record.u, hit_record.v, &hit_record.p);

    let maybe_reflection = hit_record.material.scatter(ray, &hit_record);
    match maybe_reflection {
        //todo rename colour to attenutation when i understand what that is.
        Some((reflected_ray, surface_colour)) => {
            let reflection_colour = ray_colour(reflected_ray, world, depth - 1, background);

            return emitted.add(&Colour {
                x: surface_colour.x * reflection_colour.x,
                y: surface_colour.y * reflection_colour.y,
                z: surface_colour.z * reflection_colour.z,
            });
        }

        None => {
            return emitted;
        }
    }
}

type Colour = Vec3;
type Point3 = Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn add(&self, other: &Vec3) -> Vec3 {
        return Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }

    fn subtract(&self, other: &Vec3) -> Vec3 {
        return Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        };
    }

    fn multiply(&self, scalar: f32) -> Vec3 {
        return Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        };
    }

    fn divide(&self, scalar: f32) -> Vec3 {
        return self.multiply(1.0 / scalar);
    }

    fn length_squared(&self) -> f32 {
        return self.x.powi(2) + self.y.powi(2) + self.z.powi(2);
    }

    fn length(&self) -> f32 {
        return self.length_squared().sqrt();
    }

    fn unit_vector(&self) -> Vec3 {
        return self.divide(self.length());
    }

    //for printing colours only
    // fn x_for_printing(&self) -> String {
    //     // return ((255.999 * self.x) as i32).to_string();
    //     return self.safe_colour_print(self.x);
    // }
    // fn y_for_printing(&self) -> String {
    //     // return ((255.999 * self.y) as i32).to_string();
    //     return self.safe_colour_print(self.y);
    // }
    // fn z_for_printing(&self) -> String {
    //     // return ((255.999 * self.z) as i32).to_string();
    //     return self.safe_colour_print(self.z);
    // }

    // // Check that printing is outputing valid integer.
    // // Should be "Static" equivalent, but I'm on a plane and can't look it up.
    // fn safe_colour_print(&self, coord: f32) -> String {
    //     let as_i32 = (255.999 * coord) as i32;
    //     assert!(as_i32 >= 0, "failed to print because as_i32 is {}", as_i32);
    //     assert!(as_i32 < 256, "failed to print because as_i32 is {}", as_i32);
    //     return as_i32.to_string();
    // }

    fn near_zero(&self) -> bool {
        let threshold = 1e-8;
        return self.x < threshold && self.y < threshold && self.z < threshold;
    }
}

//todo have this or static method above but not both.
fn safe_colour_print(coord: f32) -> String {
    let as_i32 = clamp(255.999 * coord, 0.0, 255.0) as i32;
    assert!(as_i32 >= 0, "failed to print because as_i32 is {}", as_i32);
    assert!(as_i32 < 256, "failed to print because as_i32 is {}", as_i32);
    return as_i32.to_string();
}

pub struct Ray {
    origin: Point3,
    direction: Point3,
}

impl Ray {
    fn at(&self, t: f32) -> Point3 {
        return self.origin.add(&self.direction.multiply(t));
    }
}

fn hit_record_with_norml<'a>(
    p: Vec3,
    t: f32,
    material: &Material,
    ray: &Ray,
    outward_normal: Vec3,
    u: f32,
    v: f32,
) -> HitRecord {
    let is_front_face = dot(ray.direction, outward_normal) < 0.0;
    let normal = if is_front_face {
        outward_normal
    } else {
        outward_normal.multiply(-1.0)
    };
    return HitRecord {
        p,
        normal,
        t,
        is_front_face,
        material: (*material).clone(),
        u,
        v,
    };
}

#[derive(Clone)]
pub struct HitRecord {
    // P is the point of hitting
    p: Point3,
    normal: Vec3,
    t: f32,
    is_front_face: bool,
    material: Material,
    // surface co-ordinates of the hit point
    // This seems a slightly odd representation, see section
    // 4.2 of The Next Week for details.
    u: f32,
    v: f32,
}

// impl HitRecord {
//     fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
//         // If the ray direciton and outward normal are in opposite direction (dot
//         // product < 0), then the ray is hitting the outside.
//         self.is_front_face = dot(ray.direction, outward_normal) < 0.0;
//         self.normal = if self.is_front_face {
//             outward_normal
//         } else {
//             outward_normal.multiply(-1.0)
//         };
//     }
// }

#[derive(Clone, Copy)]
struct HittableList<'a> {
    objects: &'a Vec<HittableObject<'a>>,
}

impl HittableList<'_> {
    // impl HittableObject for HittableList<'_> {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut temp_record = None;
        let mut closest_so_far = t_max;

        for scene_object in self.objects {
            let this_hit_record = scene_object.hit(ray, t_min, closest_so_far);
            match this_hit_record {
                Some(hit_record) => {
                    closest_so_far = hit_record.t;
                    temp_record = Some(hit_record);
                }
                None => (),
            }
        }
        return temp_record;
    }

    fn bounding_box(&self, time0: f32, time1: f32) -> Option<BoundingBox> {
        let maybe_boxes = self
            .objects
            .into_iter()
            .map(|obj| obj.bounding_box(time0, time1));
        let boxes_if_zero_nones = maybe_boxes.collect::<Option<Vec<BoundingBox>>>()?;
        return boxes_if_zero_nones
            .into_iter()
            .reduce(|box1, box2| surrounding_box(&box1, &box2));
    }
}

fn surrounding_box(box_0: &BoundingBox, box_1: &BoundingBox) -> BoundingBox {
    let small = Point3 {
        x: box_0.minimum.x.min(box_1.minimum.x),
        y: box_0.minimum.y.min(box_1.minimum.y),
        z: box_0.minimum.z.min(box_1.minimum.z),
    };
    let large = Point3 {
        x: box_0.maximum.x.max(box_1.maximum.x),
        y: box_0.maximum.y.max(box_1.maximum.y),
        z: box_0.maximum.z.max(box_1.maximum.z),
    };
    return BoundingBox {
        minimum: small,
        maximum: large,
    };
}

#[derive(Clone)]
struct Sphere {
    centre: Point3,
    radius: f32,
    material: Material,
}

impl Sphere {
    /// See code on hit_sphere for logic and and quadratic optimisation
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let origin_centre = ray.origin.subtract(&self.centre);
        let a = dot(ray.direction, ray.direction);
        let half_b = dot(origin_centre, ray.direction);
        // let b = 2.0 * dot(origin_centre, ray.direction);
        let c = dot(origin_centre, origin_centre) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        // return discriminant > 0.0;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();

        let mut root = (-half_b - sqrt_discriminant) / a;
        if root < t_min || root > t_max {
            // First root invalid, try again
            root = (-half_b + sqrt_discriminant) / a;
            if root < t_min || root > t_max {
                // Other solution is a miss too
                return None;
            }
        }
        let p = ray.at(root);
        let outward_normal = p.subtract(&self.centre).divide(self.radius);
        let (u, v) = Sphere::get_sphere_uv(&p);
        return Some(hit_record_with_norml(
            p,
            root,
            &self.material,
            ray,
            outward_normal,
            u,
            v,
        ));
    }

    fn bounding_box(&self, _time0: f32, _time1: f32) -> Option<BoundingBox> {
        let radius_box = Vec3 {
            x: self.radius,
            y: self.radius,
            z: self.radius,
        };
        let output = BoundingBox {
            minimum: self.centre.subtract(&radius_box),
            maximum: self.centre.add(&radius_box),
        };
        return Some(output);
    }

    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    fn get_sphere_uv(point: &Point3) -> (f32, f32) {
        //todo should test this
        let pi = std::f32::consts::PI;
        let theta = -point.y.acos();
        let phi = -point.z.atan2(point.x) + pi;

        return (phi / (2.0 * pi), theta / pi);
    }
}

struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,

    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = random_in_unit_disk().multiply(self.lens_radius);
        let offset = self.u.multiply(rd.x).add(&&self.v.multiply(rd.y));

        let ray_direction = self
            .lower_left_corner
            .add(&self.horizontal.multiply(s as f32))
            .add(&self.vertical.multiply(t as f32))
            .subtract(&self.origin)
            .subtract(&offset);

        let ray = Ray {
            origin: self.origin.add(&offset),
            direction: ray_direction,
        };
        return ray;
    }
}

// look_from is the position of the camera.
// look_at is where the camera is pointing to (note: it is a point, rather than a direction)
// v_up is an "up" vector for the camera. look_from and look_at only specify
// the camera within a rotation (think of tilting your head while fixing your gaze)
fn build_camera(
    look_from: Point3,
    look_at: Point3,
    v_up: Vec3,
    vertical_field_of_view: f32,
    aspect_ratio: f32,
    aperture: f32,
    focus_dist: f32,
) -> Camera {
    let theta = degrees_to_radians(vertical_field_of_view);
    let h = (theta / 2.0).tan();
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    // We're going to build a basis for the vector space.
    // w is s.t. -w is the camera angle.
    let w = look_from.subtract(&look_at).unit_vector();
    let u = cross(v_up, w);
    let v = cross(w, u);

    let origin = look_from;
    let horizontal = u.multiply(viewport_width).multiply(focus_dist);
    let vertical = v.multiply(viewport_height).multiply(focus_dist);

    let half_plane = horizontal.multiply(0.5).add(&vertical.multiply(0.5));
    let lower_left_corner = origin
        .subtract(&half_plane)
        .subtract(&&w.multiply(focus_dist));

    let lens_radius = aperture / 2.0;

    return Camera {
        origin,
        horizontal,
        vertical,
        lower_left_corner,

        u,
        v,
        lens_radius,
    };
}

fn degrees_to_radians(degrees: f32) -> f32 {
    return degrees * std::f32::consts::PI / 180.0;
}

fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        return min;
    }
    if x > max {
        return max;
    }
    return x;
}

fn random_unit() -> f32 {
    // let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(230);
    return rng.gen_range(0.0..1.0);
}

fn random_vec3() -> Vec3 {
    let mut rng = rand::thread_rng();
    // let mut rng = StdRng::seed_from_u64(230);

    return Vec3 {
        x: rng.gen_range(0.0..1.0),
        y: rng.gen_range(0.0..1.0),
        z: rng.gen_range(0.0..1.0),
    };
}

fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = random_vec3();
        let q = Vec3 {
            x: p.x,
            y: p.y,
            z: 0.0,
        };
        if q.length_squared() >= 1.0 {
            continue;
        }
        return q;
    }
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = random_vec3();
        if p.length_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

fn random_unit_vector() -> Vec3 {
    return random_in_unit_sphere().unit_vector();
}

#[derive(Clone, Copy)]
pub struct BoundingBox {
    minimum: Point3,
    maximum: Point3,
}

/// Core Idea:
/// Rays are given as a function of time, P(t) = Origin + direction*t
/// Given an interval (x_0, x_1) on a single dimension, you can find the time
/// interval that the ray crosses that interval.
/// E.g. Origin_x + direction_x*t_0 = x_0 can be solved for x_0 as
///
/// t_0 = (x_0 - Origin_x) / direction_x.
///
/// And likewise solving for t_1, the exit time.
///
/// This gives an interval (t0, t1) for the intersection on one axis.
/// Taking the intersection of these intervals on 3 axis will give you
/// the interval the box is hit (if any).
impl BoundingBox {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        // todo divisions by zero - actually maybe these work out fine with infinities?
        let x_interval = self.x_hit_interval(ray);
        let y_interval = self.y_hit_interval(ray);
        let z_interval = self.z_hit_interval(ray);

        // Now we have three time intervals where the ray crosses the bounding box, one for each axis.
        // The box is hit IFF these boxes have non-zero intersection. That is, if the highest
        // min is lower than the lowest max
        let max_of_mins = t_min.max(x_interval.0.max(y_interval.0.max(z_interval.0)));
        let min_of_maxes = t_max.min(x_interval.1.min(y_interval.1.min(z_interval.1)));

        return max_of_mins < min_of_maxes;
    }
    fn x_hit_interval(&self, ray: &Ray) -> (f32, f32) {
        let x_hit_1 = (self.minimum.x - ray.origin.x) / ray.direction.x;
        let x_hit_2 = (self.maximum.x - ray.origin.x) / ray.direction.x;
        if x_hit_1 > x_hit_2 {
            return (x_hit_2, x_hit_1);
        } else {
            return (x_hit_1, x_hit_2);
        }
    }

    fn y_hit_interval(&self, ray: &Ray) -> (f32, f32) {
        let y_hit_1 = (self.minimum.y - ray.origin.y) / ray.direction.y;
        let y_hit_2 = (self.maximum.y - ray.origin.y) / ray.direction.y;
        if y_hit_1 > y_hit_2 {
            return (y_hit_2, y_hit_1);
        } else {
            return (y_hit_1, y_hit_2);
        }
    }

    fn z_hit_interval(&self, ray: &Ray) -> (f32, f32) {
        let z_hit_1 = (self.minimum.z - ray.origin.z) / ray.direction.z;
        let z_hit_2 = (self.maximum.z - ray.origin.z) / ray.direction.z;
        if z_hit_1 > z_hit_2 {
            return (z_hit_2, z_hit_1);
        } else {
            return (z_hit_1, z_hit_2);
        }
    }
}

#[derive(Clone)]
struct BVH<'a> {
    left: Box<HittableObject<'a>>,
    right: Box<HittableObject<'a>>,
    bounding_box: BoundingBox,
}

#[derive(Clone)]
enum HittableObject<'a> {
    Sphere(Sphere),
    HittableList(HittableList<'a>),
    BVH(BVH<'a>),
    Rectangle(Rectangle),
}

impl<'a> HittableObject<'a> {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        match self {
            HittableObject::Sphere(sphere) => sphere.hit(ray, t_min, t_max),
            HittableObject::BVH(bvh) => bvh.hit(ray, t_min, t_max),
            HittableObject::HittableList(hittable_list) => hittable_list.hit(ray, t_min, t_max),
            HittableObject::Rectangle(rectangle) => rectangle.hit(ray, t_min, t_max),
        }
    }
    fn bounding_box(&self, time0: f32, time1: f32) -> Option<BoundingBox> {
        match self {
            HittableObject::Sphere(sphere) => sphere.bounding_box(time0, time1),
            HittableObject::BVH(bvh) => bvh.bounding_box(time0, time1),
            HittableObject::HittableList(hittable_list) => hittable_list.bounding_box(time0, time1),
            HittableObject::Rectangle(rectangle) => rectangle.bounding_box(time0, time1),
        }
    }
}

impl<'a> BVH<'a> {
    fn build_bvh(objects: &'a mut [HittableObject], time0: f32, time1: f32) -> BVH<'a> {
        let comparator = get_random_axis_comparator();
        let left: HittableObject;
        let right: HittableObject;

        let object_span = objects.len();
        if object_span == 1 {
            left = objects[0].clone();
            right = objects[0].clone();
        } else if object_span == 2 {
            if comparator(&&objects[0], &&objects[1]) == Ordering::Less {
                left = objects[0].clone();
                right = objects[1].clone();
            } else {
                left = objects[1].clone();
                right = objects[0].clone();
            }
        } else {
            objects.sort_by(comparator);
            let (new_head, new_tail) = objects.split_at_mut(objects.len() / 2);
            left = HittableObject::BVH(BVH::build_bvh(new_head, time0, time1));
            right = HittableObject::BVH(BVH::build_bvh(new_tail, time0, time1));
        }

        let left_bb = left.bounding_box(time0, time1);
        let right_bb = right.bounding_box(time0, time1);
        if left_bb.is_none() || right_bb.is_none() {
            unreachable!("Can't construct a BVH if the objects aren't bounded!");
        }
        let left_box = Box::new(left);
        let right_box = Box::new(right);

        return BVH {
            left: left_box,
            right: right_box,
            bounding_box: surrounding_box(&left_bb.unwrap(), &right_bb.unwrap()),
        };
    }
}

fn get_random_axis_comparator() -> fn(&HittableObject, &HittableObject) -> Ordering {
    let noise: i32 = rand::random();
    if noise % 3 == 0 {
        return box_compare_x;
    } else if noise % 3 == 1 {
        return box_compare_y;
    } else {
        return box_compare_z;
    }
}

fn box_compare_x(box0: &HittableObject, box1: &HittableObject) -> Ordering {
    return box_compare(box0, box1, 0);
}
fn box_compare_y(box0: &HittableObject, box1: &HittableObject) -> Ordering {
    return box_compare(box0, box1, 1);
}
fn box_compare_z(box0: &HittableObject, box1: &HittableObject) -> Ordering {
    return box_compare(box0, box1, 2);
}

fn box_compare(box0: &HittableObject, box1: &HittableObject, axis: i32) -> Ordering {
    let box_a = box0.bounding_box(0.0, 0.0).unwrap();
    let box_b = box1.bounding_box(0.0, 0.0).unwrap();
    if axis == 0 {
        return if box_a.minimum.x < box_b.minimum.x {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    } else if axis == 1 {
        return if box_a.minimum.y < box_b.minimum.y {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    } else if axis == 2 {
        return if box_a.minimum.z < box_b.minimum.z {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    } else {
        unreachable!()
    }
}

impl BVH<'_> {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if !self.bounding_box.hit(ray, t_min, t_max) {
            return None;
        }
        let hit_left = self.left.hit(ray, t_min, t_max);
        match hit_left {
            None => return self.right.hit(ray, t_min, t_max),
            Some(ref hit_on_left) => return self.right.hit(ray, t_min, hit_on_left.t).or(hit_left),
        }
    }

    fn bounding_box(&self, _time0: f32, _time1: f32) -> Option<BoundingBox> {
        return Some(self.bounding_box);
    }
}
